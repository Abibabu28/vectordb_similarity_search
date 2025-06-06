# Privacy-Preserving Patient Similarity Search in Liver Transplant Research
# Implementation using Federated Learning, Vector Databases & Flower
# Modified to include liver transplant outcomes

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Metrics
import logging
import json
from datetime import datetime, timedelta
import hashlib
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== PATIENT EMBEDDING MODEL ==================

class PatientEmbeddingModel(nn.Module):
    """Neural network to generate patient embeddings for similarity search"""
    
    def __init__(self, input_dim: int = 20, embedding_dim: int = 128, hidden_dims: List[int] = [256, 128]):
        super(PatientEmbeddingModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# ================== ENHANCED SYNTHETIC PATIENT DATA GENERATOR ==================

class SyntheticPatientDataGenerator:
    """Generate synthetic liver transplant patient data with transplant outcomes"""
    
    @staticmethod
    def generate_patient_features(n_patients: int = 1000) -> pd.DataFrame:
        """Generate synthetic UNOS-like patient features with transplant outcomes"""
        np.random.seed(42)
        
        # Simulate key liver transplant features
        data = {
            'age': np.random.normal(55, 15, n_patients).clip(18, 80),
            'meld_score': np.random.exponential(15, n_patients).clip(6, 40),
            'bmi': np.random.normal(27, 5, n_patients).clip(18, 45),
            'creatinine': np.random.exponential(1.2, n_patients).clip(0.5, 8),
            'bilirubin': np.random.exponential(5, n_patients).clip(0.3, 50),
            'inr': np.random.exponential(1.8, n_patients).clip(0.8, 6),
            'sodium': np.random.normal(138, 5, n_patients).clip(125, 150),
            'albumin': np.random.normal(3.2, 0.8, n_patients).clip(1.5, 5),
            'dialysis': np.random.binomial(1, 0.15, n_patients),
            'ascites': np.random.binomial(1, 0.4, n_patients),
            'encephalopathy': np.random.binomial(1, 0.25, n_patients),
            'diabetes': np.random.binomial(1, 0.3, n_patients),
            'hypertension': np.random.binomial(1, 0.45, n_patients),
            'etiology_alcohol': np.random.binomial(1, 0.3, n_patients),
            'etiology_nash': np.random.binomial(1, 0.25, n_patients),
            'etiology_hcv': np.random.binomial(1, 0.2, n_patients),
            'etiology_other': np.random.binomial(1, 0.25, n_patients),
            'blood_type_o': np.random.binomial(1, 0.45, n_patients),
            'blood_type_a': np.random.binomial(1, 0.4, n_patients),
            'blood_type_b': np.random.binomial(1, 0.15, n_patients),
        }
        
        # Add patient IDs
        data['patient_id'] = [f"PT_{i:06d}" for i in range(n_patients)]
        
        df = pd.DataFrame(data)
        
        # Generate transplant outcomes based on realistic probabilities
        df = SyntheticPatientDataGenerator._generate_transplant_outcomes(df)
        
        return df
    
    @staticmethod
    def _generate_transplant_outcomes(df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic transplant outcomes based on patient characteristics"""
        
        # Calculate transplant probability based on MELD score and other factors
        # Higher MELD = higher priority = higher chance of transplant
        meld_factor = (df['meld_score'] - 6) / (40 - 6)  # Normalize MELD to 0-1
        age_factor = 1 - ((df['age'] - 18) / (80 - 18)) * 0.3  # Younger patients more likely
        
        # Adjust probability based on medical conditions
        dialysis_penalty = df['dialysis'] * 0.2  # Dialysis patients have complications
        diabetes_penalty = df['diabetes'] * 0.1
        
        # Base transplant probability (realistic rates)
        base_prob = 0.25  # About 25% of waitlist patients receive transplants
        
        transplant_prob = (base_prob + meld_factor * 0.4 + age_factor * 0.1 
                          - dialysis_penalty - diabetes_penalty).clip(0.05, 0.8)
        
        # Generate transplant status
        df['received_transplant'] = np.random.binomial(1, transplant_prob, len(df))
        
        # For patients who received transplants, generate additional outcome data
        transplanted_mask = df['received_transplant'] == 1
        n_transplanted = transplanted_mask.sum()
        
        if n_transplanted > 0:
            # Time to transplant (days on waitlist)
            df.loc[transplanted_mask, 'days_to_transplant'] = np.random.exponential(120, n_transplanted).clip(1, 1000)
            
            # Transplant success (1-year survival)
            # Success rate depends on age, MELD score, and comorbidities
            success_base_rate = 0.85  # 85% 1-year survival rate
            age_penalty = (df.loc[transplanted_mask, 'age'] - 50) / 100  # Older = higher risk
            meld_penalty = (df.loc[transplanted_mask, 'meld_score'] - 15) / 100  # Higher MELD = higher risk
            comorbidity_penalty = (df.loc[transplanted_mask, 'diabetes'] + 
                                 df.loc[transplanted_mask, 'dialysis']) * 0.05
            
            success_prob = (success_base_rate - age_penalty - meld_penalty - comorbidity_penalty).clip(0.3, 0.95)
            df.loc[transplanted_mask, 'transplant_success'] = np.random.binomial(1, success_prob, n_transplanted)
            
            # Generate transplant dates (within last 5 years)
            base_date = datetime.now() - timedelta(days=5*365)
            random_days = np.random.randint(0, 5*365, n_transplanted)
            df.loc[transplanted_mask, 'transplant_date'] = [
                (base_date + timedelta(days=int(days))).strftime('%Y-%m-%d') 
                for days in random_days
            ]
            
            # Post-transplant follow-up time (days since transplant)
            df.loc[transplanted_mask, 'follow_up_days'] = np.random.exponential(400, n_transplanted).clip(30, 1800)
        
        # For patients who didn't receive transplant
        not_transplanted_mask = df['received_transplant'] == 0
        n_not_transplanted = not_transplanted_mask.sum()
        
        if n_not_transplanted > 0:
            # Time on waitlist (for those still waiting or removed)
            df.loc[not_transplanted_mask, 'days_on_waitlist'] = np.random.exponential(200, n_not_transplanted).clip(1, 2000)
            
            # Waitlist status: 0=Still active, 1=Removed (too sick), 2=Removed (improved), 3=Deceased
            waitlist_status_probs = [0.6, 0.2, 0.1, 0.1]  # Probabilities for each status
            df.loc[not_transplanted_mask, 'waitlist_status'] = np.random.choice(
                [0, 1, 2, 3], n_not_transplanted, p=waitlist_status_probs
            )
        
        # Fill NaN values for non-applicable fields
        df['days_to_transplant'] = df['days_to_transplant'].fillna(0)
        df['transplant_success'] = df['transplant_success'].fillna(0)
        df['transplant_date'] = df['transplant_date'].fillna('N/A')
        df['follow_up_days'] = df['follow_up_days'].fillna(0)
        df['days_on_waitlist'] = df['days_on_waitlist'].fillna(0)
        df['waitlist_status'] = df['waitlist_status'].fillna(0)
        
        return df

# ================== FLOWER CLIENT IMPLEMENTATION ==================

class PatientSimilarityClient(fl.client.NumPyClient):
    """Flower client for federated learning of patient embeddings"""
    
    def __init__(self, hospital_id: str, patient_data: pd.DataFrame):
        self.hospital_id = hospital_id
        self.patient_data = patient_data
        self.model = PatientEmbeddingModel()
        self.criterion = nn.MSELoss()  # For autoencoder-like training
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Prepare data
        self.X = self._prepare_features()
        
        logger.info(f"Initialized client for {hospital_id} with {len(patient_data)} patients")
        logger.info(f"  - Transplanted: {patient_data['received_transplant'].sum()}")
        logger.info(f"  - Still on waitlist: {(patient_data['received_transplant'] == 0).sum()}")
    
    def _prepare_features(self) -> torch.Tensor:
        """Prepare patient features for training"""
        feature_cols = [
            'age', 'meld_score', 'bmi', 'creatinine', 'bilirubin', 'inr',
            'sodium', 'albumin', 'dialysis', 'ascites', 'encephalopathy',
            'diabetes', 'hypertension', 'etiology_alcohol', 'etiology_nash',
            'etiology_hcv', 'etiology_other', 'blood_type_o', 'blood_type_a',
            'blood_type_b'
        ]
        X = self.patient_data[feature_cols].values.astype(np.float32)
        
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        return torch.FloatTensor(X)
    
    def get_parameters(self, config):
        """Return model parameters"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model locally"""
        self.set_parameters(parameters)
        
        # Training loop (autoencoder approach for unsupervised embedding learning)
        self.model.train()
        epoch_losses = []
        
        batch_size = min(32, len(self.X))
        n_batches = len(self.X) // batch_size
        
        for epoch in range(config.get("local_epochs", 5)):
            epoch_loss = 0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.X))
                batch_x = self.X[start_idx:end_idx]
                
                self.optimizer.zero_grad()
                
                # Generate embeddings
                embeddings = self.model(batch_x)
                
                # Reconstruction loss (simplified autoencoder objective)
                reconstructed = torch.matmul(embeddings, embeddings.T)
                target = torch.matmul(batch_x, batch_x.T)
                loss = self.criterion(reconstructed, target)
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_losses.append(epoch_loss / n_batches)
        
        logger.info(f"{self.hospital_id}: Training completed. Final loss: {epoch_losses[-1]:.4f}")
        
        return self.get_parameters(config={}), len(self.X), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model"""
        self.set_parameters(parameters)
        self.model.eval()
        
        with torch.no_grad():
            embeddings = self.model(self.X)
            # Simple evaluation metric: embedding variance (higher = more diverse representations)
            embedding_var = torch.var(embeddings).item()
        
        return float(embedding_var), len(self.X), {"embedding_variance": embedding_var}

# ================== ENHANCED IN-MEMORY VECTOR STORAGE ==================

class HospitalVectorStorage:
    """Manages patient embeddings with transplant outcome information"""
    
    def __init__(self, hospital_id: str):
        self.hospital_id = hospital_id
        self.patient_embeddings = {}  # patient_id -> embedding
        self.patient_metadata = {}    # patient_id -> metadata (including transplant info)
        self.embeddings_matrix = None # numpy matrix for efficient similarity search
        self.patient_ids_list = []    # ordered list of patient IDs
        
        logger.info(f"In-memory vector storage initialized for {hospital_id}")
    
    def store_patient_embeddings(self, patient_ids: List[str], embeddings: np.ndarray, 
                               metadata: List[Dict]):
        """Store patient embeddings with transplant outcome metadata"""
        
        # Store individual embeddings and metadata
        for i, patient_id in enumerate(patient_ids):
            self.patient_embeddings[patient_id] = embeddings[i]
            self.patient_metadata[patient_id] = metadata[i]
        
        # Update matrix for efficient similarity search
        self.patient_ids_list = list(self.patient_embeddings.keys())
        self.embeddings_matrix = np.array([self.patient_embeddings[pid] for pid in self.patient_ids_list])
        
        # Log transplant statistics
        transplanted_count = sum(1 for meta in metadata if meta.get('received_transplant'))
        logger.info(f"{self.hospital_id}: Stored {len(patient_ids)} patient embeddings")
        logger.info(f"  - {transplanted_count} patients received transplants")
        logger.info(f"  - {len(patient_ids) - transplanted_count} patients did not receive transplants")
    
    def search_similar_patients(self, query_embedding: np.ndarray, top_k: int = 10) -> Dict:
        """Find most similar patients with transplant outcome information"""
        
        if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
            return {'patient_ids': [], 'distances': [], 'metadata': []}
        
        # Compute cosine similarity
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        # Get top-k most similar patients
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = {
            'patient_ids': [self.patient_ids_list[i] for i in top_indices],
            'distances': [1 - similarities[i] for i in top_indices],  # Convert to distance
            'metadata': [self.patient_metadata[self.patient_ids_list[i]] for i in top_indices]
        }
        
        return results

# Create alias for compatibility
HospitalVectorDB = HospitalVectorStorage

# ================== ENHANCED MPC SIMULATION ==================

class SecureMultiPartyComputation:
    """Simplified MPC simulation for secure similarity computation with transplant outcomes"""
    
    @staticmethod
    def secure_similarity_search(query_embedding: np.ndarray, 
                               hospital_dbs: List[HospitalVectorDB],
                               top_k: int = 10) -> Dict:
        """
        Simulate secure similarity search across hospitals with transplant outcome analysis
        """
        
        all_results = []
        
        for hospital_db in hospital_dbs:
            # Get local similarities
            local_results = hospital_db.search_similar_patients(query_embedding, top_k)
            
            for i, patient_id in enumerate(local_results['patient_ids']):
                metadata = local_results['metadata'][i]
                
                all_results.append({
                    'patient_id': patient_id,
                    'similarity': 1 - local_results['distances'][i],
                    'hospital': hospital_db.hospital_id,
                    'metadata': metadata,
                    # Extract key transplant information for easy access
                    'received_transplant': metadata.get('received_transplant', False),
                    'transplant_success': metadata.get('transplant_success', False),
                    'days_to_transplant': metadata.get('days_to_transplant', 0),
                    'transplant_date': metadata.get('transplant_date', 'N/A'),
                    'waitlist_status': metadata.get('waitlist_status', 0)
                })
        
        # Sort by similarity and return top results
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Calculate transplant outcome statistics
        top_results = all_results[:top_k]
        transplant_stats = SecureMultiPartyComputation._calculate_transplant_statistics(top_results)
        
        return {
            'top_similar_patients': top_results,
            'total_searched': len(all_results),
            'transplant_statistics': transplant_stats
        }
    
    @staticmethod
    def _calculate_transplant_statistics(results: List[Dict]) -> Dict:
        """Calculate transplant outcome statistics for similar patients"""
        
        if not results:
            return {}
        
        total_patients = len(results)
        transplanted_patients = [r for r in results if r['received_transplant']]
        not_transplanted = [r for r in results if not r['received_transplant']]
        
        stats = {
            'total_similar_patients': total_patients,
            'transplanted_count': len(transplanted_patients),
            'not_transplanted_count': len(not_transplanted),
            'transplant_rate': len(transplanted_patients) / total_patients if total_patients > 0 else 0,
        }
        
        if transplanted_patients:
            successful_transplants = [r for r in transplanted_patients if r['transplant_success']]
            avg_wait_time = np.mean([r['days_to_transplant'] for r in transplanted_patients])
            
            stats.update({
                'successful_transplants': len(successful_transplants),
                'transplant_success_rate': len(successful_transplants) / len(transplanted_patients),
                'average_wait_time_days': avg_wait_time,
                'average_wait_time_months': avg_wait_time / 30.44  # Average days per month
            })
        
        if not_transplanted:
            still_active = len([r for r in not_transplanted if r['waitlist_status'] == 0])
            removed_sick = len([r for r in not_transplanted if r['waitlist_status'] == 1])
            removed_improved = len([r for r in not_transplanted if r['waitlist_status'] == 2])
            deceased = len([r for r in not_transplanted if r['waitlist_status'] == 3])
            
            stats.update({
                'still_on_waitlist': still_active,
                'removed_too_sick': removed_sick,
                'removed_improved': removed_improved,
                'deceased_on_waitlist': deceased
            })
        
        return stats

# ================== ENHANCED MAIN SYSTEM ORCHESTRATOR ==================

class PrivacyPreservingPatientSearch:
    """Main system orchestrating the entire pipeline with transplant outcomes"""
    
    def __init__(self):
        self.hospitals = {}
        self.hospital_dbs = {}
        self.global_model = PatientEmbeddingModel()
        self.mpc = SecureMultiPartyComputation()
        
    def setup_hospitals(self, hospital_configs: Dict[str, int]):
        """Setup hospitals with synthetic patient data including transplant outcomes"""
        
        for hospital_id, n_patients in hospital_configs.items():
            # Generate synthetic patient data with transplant outcomes
            patient_data = SyntheticPatientDataGenerator.generate_patient_features(n_patients)
            
            self.hospitals[hospital_id] = {
                'data': patient_data,
                'client': PatientSimilarityClient(hospital_id, patient_data)
            }
            
            # Initialize vector database for each hospital
            self.hospital_dbs[hospital_id] = HospitalVectorDB(hospital_id)
            
        logger.info(f"Setup complete for {len(hospital_configs)} hospitals")
    
    def run_federated_training(self, rounds: int = 3, local_epochs: int = 5):
        """Simulate federated learning training process"""
        
        logger.info(f"Starting federated training for {rounds} rounds")
        
        # Initialize global model parameters
        global_params = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        
        for round_num in range(rounds):
            logger.info(f"Round {round_num + 1}/{rounds}")
            
            round_params = []
            round_metrics = []
            
            # Each hospital trains locally
            for hospital_id, hospital_info in self.hospitals.items():
                client = hospital_info['client']
                
                # Local training
                params, n_samples, metrics = client.fit(
                    global_params, 
                    {"local_epochs": local_epochs}
                )
                
                round_params.append((params, n_samples))
                round_metrics.append(metrics)
            
            # Aggregate parameters (FedAvg)
            global_params = self._federated_averaging(round_params)
            
            # Update global model
            params_dict = zip(self.global_model.state_dict().keys(), global_params)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.global_model.load_state_dict(state_dict, strict=True)
            
            logger.info(f"Round {round_num + 1} completed")
        
        logger.info("Federated training completed!")
        return self.global_model
    
    def _federated_averaging(self, params_and_samples: List[Tuple]) -> List[np.ndarray]:
        """Implement FedAvg aggregation"""
        
        total_samples = sum(n_samples for _, n_samples in params_and_samples)
        
        # Initialize aggregated parameters
        aggregated_params = None
        
        for params, n_samples in params_and_samples:
            weight = n_samples / total_samples
            
            if aggregated_params is None:
                aggregated_params = [weight * param for param in params]
            else:
                for i, param in enumerate(params):
                    aggregated_params[i] += weight * param
        
        return aggregated_params
    
    def generate_and_store_embeddings(self):
        """Generate embeddings for all patients with transplant outcome metadata"""
        
        self.global_model.eval()
        
        for hospital_id, hospital_info in self.hospitals.items():
            patient_data = hospital_info['data']
            client = hospital_info['client']
            
            # Generate embeddings using the trained global model
            with torch.no_grad():
                embeddings = self.global_model(client.X).numpy()
            
            # Prepare enhanced metadata including transplant outcomes
            metadata = []
            for _, row in patient_data.iterrows():
                meta = {
                    'age': float(row['age']),
                    'meld_score': float(row['meld_score']),
                    'bmi': float(row['bmi']),
                    'hospital': hospital_id,
                    'timestamp': datetime.now().isoformat(),
                    # Transplant outcome information
                    'received_transplant': bool(row['received_transplant']),
                    'transplant_success': bool(row['transplant_success']) if row['received_transplant'] else False,
                    'days_to_transplant': float(row['days_to_transplant']) if row['received_transplant'] else 0,
                    'transplant_date': str(row['transplant_date']),
                    'follow_up_days': float(row['follow_up_days']) if row['received_transplant'] else 0,
                    'days_on_waitlist': float(row['days_on_waitlist']) if not row['received_transplant'] else 0,
                    'waitlist_status': int(row['waitlist_status']) if not row['received_transplant'] else 0,
                    # Clinical details for context
                    'creatinine': float(row['creatinine']),
                    'bilirubin': float(row['bilirubin']),
                    'dialysis': bool(row['dialysis']),
                    'diabetes': bool(row['diabetes'])
                }
                metadata.append(meta)
            
            # Store in vector database
            patient_ids = patient_data['patient_id'].tolist()
            self.hospital_dbs[hospital_id].store_patient_embeddings(
                patient_ids, embeddings, metadata
            )
        
        logger.info("All patient embeddings with transplant outcomes generated and stored!")
    
    def search_similar_patients(self, query_patient_data: Dict, top_k: int = 10) -> Dict:
        """Search for similar patients with transplant outcome analysis"""
        
        # Convert query patient data to features
        query_features = self._prepare_query_features(query_patient_data)
        
        # Generate embedding for query patient
        self.global_model.eval()
        with torch.no_grad():
            query_embedding = self.global_model(query_features).numpy().squeeze()
        
        # Perform secure similarity search across hospitals
        results = self.mpc.secure_similarity_search(
            query_embedding, 
            list(self.hospital_dbs.values()), 
            top_k
        )
        
        return results
    
    def _prepare_query_features(self, query_data: Dict) -> torch.Tensor:
        """Prepare query patient features for embedding generation"""
        
        # Expected feature order (should match training data)
        feature_order = [
            'age', 'meld_score', 'bmi', 'creatinine', 'bilirubin', 'inr',
            'sodium', 'albumin', 'dialysis', 'ascites', 'encephalopathy',
            'diabetes', 'hypertension', 'etiology_alcohol', 'etiology_nash',
            'etiology_hcv', 'etiology_other', 'blood_type_o', 'blood_type_a',
            'blood_type_b'
        ]
        
        features = np.array([query_data.get(feat, 0) for feat in feature_order], dtype=np.float32)
        
        # Normalize (in practice, you'd store normalization parameters from training)
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return torch.FloatTensor(features).unsqueeze(0)

# ================== ENHANCED DEMONSTRATION WORKFLOW ==================

def demonstrate_system():
    """Demonstrate the complete system with transplant outcome analysis"""
    
    print("=" * 90)
    print("PRIVACY-PRESERVING PATIENT SIMILARITY SEARCH WITH TRANSPLANT OUTCOMES")
    print("=" * 90)
    
    # Initialize the system
    system = PrivacyPreservingPatientSearch()
    
    # Setup hospitals with different patient populations
    hospital_configs = {
        "Hospital_A": 50000,
        "Hospital_B": 40000,
        "Hospital_C": 60000
    }
    
    print(f"\n1. Setting up {len(hospital_configs)} hospitals...")
    system.setup_hospitals(hospital_configs)
    
    # Run federated learning
    print("\n2. Running federated learning to train patient embedding model...")
    trained_model = system.run_federated_training(rounds=3, local_epochs=5)
    
    # Generate and store embeddings
    print("\n3. Generating patient embeddings with transplant outcomes...")
    system.generate_and_store_embeddings()
    
    # Demonstrate similarity search
    print("\n4. Demonstrating patient similarity search with transplant outcomes...")
    
    # Example query patient
    query_patient = {
        'age': 52,
        'meld_score': 22,
        'bmi': 28.5,
        'creatinine': 1.8,
        'bilirubin': 12.3,
        'inr': 2.1,
        'sodium': 135,
        'albumin': 2.8,
        'dialysis': 0,
        'ascites': 1,
        'encephalopathy': 1,
        'diabetes': 1,
        'hypertension': 1,
        'etiology_alcohol': 1,
        'etiology_nash': 0,
        'etiology_hcv': 0,
        'etiology_other': 0,
        'blood_type_o': 1,
        'blood_type_a': 0,
        'blood_type_b': 0
    }
    
    print(f"\nQuery Patient Profile:")
    print(f"  Age: {query_patient['age']} years")
    print(f"  MELD Score: {query_patient['meld_score']}")
    print(f"  BMI: {query_patient['bmi']}")
    print(f"  Creatinine: {query_patient['creatinine']} mg/dL")
    print(f"  Bilirubin: {query_patient['bilirubin']} mg/dL")
    print(f"  Diabetes: {'Yes' if query_patient['diabetes'] else 'No'}")
    print(f"  Alcohol-related liver disease: {'Yes' if query_patient['etiology_alcohol'] else 'No'}")
    
    # Search for similar patients
    similar_patients = system.search_similar_patients(query_patient, top_k=5)
    
    print(f"\n5. Similar Patients Found:")
    print(f"Total patients searched across hospitals: {similar_patients['total_searched']}")
    
    # Display transplant statistics
    stats = similar_patients['transplant_statistics']
    print(f"\n TRANSPLANT OUTCOME STATISTICS FOR SIMILAR PATIENTS:")
    print(f"   Total similar patients analyzed: {stats['total_similar_patients']}")
    print(f"   Patients who received transplants: {stats['transplanted_count']} ({stats['transplant_rate']:.1%})")
    print(f"   Patients who didn't receive transplants: {stats['not_transplanted_count']}")
    
    if stats['transplanted_count'] > 0:
        print(f"\n   TRANSPLANT SUCCESS METRICS:")
        print(f"   • Successful transplants: {stats['successful_transplants']}")
        print(f"   • Success rate: {stats['transplant_success_rate']:.1%}")
        print(f"   • Average wait time: {stats['average_wait_time_days']:.0f} days ({stats['average_wait_time_months']:.1f} months)")
    
    if stats['not_transplanted_count'] > 0:
        print(f"\n   WAITLIST STATUS FOR NON-TRANSPLANTED:")
        print(f"   • Still active on waitlist: {stats.get('still_on_waitlist', 0)}")
        print(f"   • Removed (too sick): {stats.get('removed_too_sick', 0)}")
        print(f"   • Removed (improved): {stats.get('removed_improved', 0)}")
        print(f"   • Deceased on waitlist: {stats.get('deceased_on_waitlist', 0)}")
    
    print(f"\nTOP 5 MOST SIMILAR PATIENTS WITH TRANSPLANT OUTCOMES:")
    print("-" * 90)
    
    for i, patient in enumerate(similar_patients['top_similar_patients']):
        print(f"\nRank {i+1}:")
        print(f"  Patient ID: {patient['patient_id']}")
        print(f"  Hospital: {patient['hospital']}")
        print(f"  Similarity Score: {patient['similarity']:.4f}")
       
        
        # Transplant outcome information
        if patient['received_transplant']:
            print(f" TRANSPLANT STATUS: RECEIVED")
            print(f"     • Transplant Date: {patient['transplant_date']}")
            print(f"     • Wait Time: {patient['days_to_transplant']:.0f} days")
            print(f"     • Success: {'Yes' if patient['transplant_success'] else ' No'}")
            print(f"     • Follow-up: {patient['metadata']['follow_up_days']:.0f} days post-transplant")
        else:
            print(f" TRANSPLANT STATUS: NOT RECEIVED")
            waitlist_status_map = {
                0: "Still active on waitlist",
                1: "Removed (too sick)",
                2: "Removed (condition improved)", 
                3: "Deceased on waitlist"
            }
            status = waitlist_status_map.get(patient['waitlist_status'], "Unknown")
            print(f"     • Current Status: {status}")
            print(f"     • Time on Waitlist: {patient['metadata']['days_on_waitlist']:.0f} days")
        
        
    
    # Generate clinical insights
    print(f"\n" + "=" * 90)
    print("🔍 CLINICAL INSIGHTS FOR QUERY PATIENT:")
    print("=" * 90)
    
    transplanted_similar = [p for p in similar_patients['top_similar_patients'] if p['received_transplant']]
    successful_transplants = [p for p in transplanted_similar if p['transplant_success']]
    
    if transplanted_similar:
        avg_wait_transplanted = np.mean([p['days_to_transplant'] for p in transplanted_similar])
        print(f" Among similar patients who received transplants:")
        print(f"   • {len(transplanted_similar)} patients received transplants")
        print(f"   • {len(successful_transplants)} had successful outcomes ({len(successful_transplants)/len(transplanted_similar):.1%})")
        print(f"   • Average wait time was {avg_wait_transplanted:.0f} days ({avg_wait_transplanted/30.44:.1f} months)")
        
        if successful_transplants:
            print(f"Success factors in similar patients:")
            avg_age_success = np.mean([p['metadata']['age'] for p in successful_transplants])
            avg_meld_success = np.mean([p['metadata']['meld_score'] for p in successful_transplants])
            print(f"   • Average age at transplant: {avg_age_success:.1f} years")
            print(f"   • Average MELD score: {avg_meld_success:.1f}")
            
            diabetes_rate_success = np.mean([p['metadata']['diabetes'] for p in successful_transplants])
            print(f"   • Diabetes prevalence: {diabetes_rate_success:.1%}")
    
    not_transplanted_similar = [p for p in similar_patients['top_similar_patients'] 
                              if not p['received_transplant']]
    
    if not_transplanted_similar:
        avg_wait_not_transplanted = np.mean([p['metadata']['days_on_waitlist'] for p in not_transplanted_similar])
        print(f"\n📉 Among similar patients who didn't receive transplants:")
        print(f"   • {len(not_transplanted_similar)} patients did not receive transplants")
        print(f"   • Average time on waitlist: {avg_wait_not_transplanted:.0f} days ({avg_wait_not_transplanted/30.44:.1f} months)")
        
        still_waiting = len([p for p in not_transplanted_similar if p['waitlist_status'] == 0])
        if still_waiting > 0:
            print(f"   • {still_waiting} are still actively waiting")
    
    
   
    
    return system

# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    # Run the complete demonstration
    system = demonstrate_system()
    
   