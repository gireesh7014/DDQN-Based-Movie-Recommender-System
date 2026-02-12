# 🎬 Reinforcement Learning Movie Recommendation System

## Deep Q-Network (DQN) Based Recommender for Cold-Start Users

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [System Architecture](#system-architecture)
5. [Step-by-Step Implementation](#step-by-step-implementation)
6. [How to Run](#how-to-run)
7. [Model Usage](#model-usage)
8. [Evaluation Metrics](#evaluation-metrics)
9. [File Structure](#file-structure)
10. [Technical Details](#technical-details)

---

## 🎯 Project Overview

This project implements an **end-to-end movie recommendation system** using **Reinforcement Learning (RL)**, specifically **Deep Q-Networks (DQN)**. Unlike traditional recommendation systems that rely on historical user data, this system can recommend movies to users in a **cold-start setting** — meaning it works even when there's no prior user interaction data.

### Key Features
- ✅ Handles cold-start users (no prior history needed)
- ✅ Learns user preferences dynamically through interaction
- ✅ Supports both simulated and real human feedback
- ✅ Uses advanced DQN with Dueling Architecture
- ✅ Comprehensive evaluation with standard IR metrics

---

## 🧩 Problem Statement

**Challenge**: Build a movie recommendation system that:
1. Works without pre-existing user profiles or ratings
2. Learns user preferences in real-time through feedback
3. Treats each user session as a sequential decision-making problem
4. Maximizes cumulative user satisfaction (reward) over time

**Why RL?**: Traditional collaborative filtering needs historical data. RL allows the system to:
- Explore different recommendations
- Learn from immediate feedback
- Adapt to changing user preferences within a session

---

## 📊 Dataset Description

**File**: `imdb_movies_2025_cleaned.csv`

| Column | Description | Example |
|--------|-------------|---------|
| `movie_name` | Title of the movie | "Sinners" |
| `genre` | Comma-separated genres | "Horror, Thriller, Drama" |
| `rating` | IMDB rating (0-10) | 7.5 |
| `description` | Plot summary text | "Twin brothers return to their hometown..." |

**Statistics**:
- Total Movies: ~7,800+
- Unique Genres: 64
- Rating Range: 1.0 - 10.0

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MOVIE RECOMMENDATION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Raw Data   │───▶│  Preprocessor │───▶│   Feature Vectors    │  │
│  │  (CSV File)  │    │  (TF-IDF+SVD) │    │  (165 dimensions)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                    │                 │
│                                                    ▼                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    RL ENVIRONMENT                             │  │
│  │  ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌───────────┐  │  │
│  │  │  State  │   │  Action   │   │ Reward  │   │   Done    │  │  │
│  │  │ (User   │   │ (Select   │   │ (User   │   │ (Episode  │  │  │
│  │  │ Profile)│   │  Movie)   │   │Feedback)│   │  End?)    │  │  │
│  │  └─────────┘   └───────────┘   └─────────┘   └───────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │         ▲                               │
│                           ▼         │                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      DQN AGENT                                │  │
│  │  ┌────────────┐    ┌────────────┐    ┌──────────────────┐   │  │
│  │  │ Q-Network  │    │  Target    │    │ Experience       │   │  │
│  │  │ (Dueling)  │    │  Network   │    │ Replay Buffer    │   │  │
│  │  └────────────┘    └────────────┘    └──────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Step-by-Step Implementation

### **Step 1: Data Loading & Exploration (Cells 1-4)**

**What happens**:
- Load the IMDB movies CSV file
- Explore data structure, missing values, distributions
- Visualize genre distribution and rating histogram

**Code Location**: Cells 1-4 in notebook

```python
df = pd.read_csv('imdb_movies_2025_cleaned.csv')
# Outputs: 7841 movies, 4 columns
```

---

### **Step 2: Feature Engineering (Cells 5-7)**

**Goal**: Convert raw movie data into numerical vectors that the neural network can process.

**Three Feature Components**:

| Feature | Method | Dimensions |
|---------|--------|------------|
| **Genre** | Multi-hot encoding | 64 |
| **Description** | TF-IDF → SVD reduction | 100 |
| **Rating** | Min-Max normalization | 1 |
| **Total** | Concatenation | **165** |

**The `MovieFeatureProcessor` Class**:
```python
processor = MovieFeatureProcessor(embedding_dim=100, n_tfidf_features=5000)
movie_features = processor.fit_transform(df)  # Shape: (7841, 165)
```

**How it works**:
1. **Genre Encoding**: "Horror, Thriller" → [0,0,1,0,...,1,0] (multi-hot)
2. **Description Embedding**: 
   - TF-IDF vectorization (5000 features)
   - SVD dimensionality reduction (100 features)
3. **Rating**: Normalize 0-10 → 0-1

---

### **Step 3: RL Environment Design (Cells 8-10)**

**The `MovieRecommendationEnv` Class**:

| Component | Description |
|-----------|-------------|
| **State** | User preference vector (165D, dynamically updated) |
| **Action** | Select movie index [0, 7840] |
| **Reward** | Feedback-based: like=+1.0, ignore=-0.5 |
| **Episode** | 10 recommendations per session |

**Reward Mapping**:
```python
FEEDBACK_REWARDS = {
    'like': 1.0,      # Strong positive
    'watch': 0.8,     # Watched the movie
    'click': 0.4,     # Showed interest
    'skip': -0.2,     # Skipped
    'ignore': -0.5    # Completely ignored
}
```

**State Update Logic**:
- If user **likes** a movie → Move user profile **towards** that movie's features
- If user **dislikes** → Move user profile **away** from that movie's features

**Simulated User**:
```python
class SimulatedUser:
    # Has a random latent preference vector
    # Generates feedback based on cosine similarity with movie features
```

---

### **Step 4: DQN Agent Implementation (Cells 11-14)**

**Components**:

#### 4.1 Experience Replay Buffer
```python
class ReplayBuffer:
    # Stores (state, action, reward, next_state, done) tuples
    # Enables batch learning from random samples
    # Breaks correlation between consecutive experiences
```

#### 4.2 Neural Network Architecture (Dueling DQN)
```
Input (165D) → Dense(512) → BN → Dropout
                    ↓
              Dense(256) → BN → Dropout
                    ↓
         ┌─────────┴─────────┐
         ↓                   ↓
    Value Stream        Advantage Stream
    Dense(128)          Dense(128)
         ↓                   ↓
    Dense(1)            Dense(7841)
         ↓                   ↓
         └────── Q(s,a) = V(s) + A(s,a) - mean(A) ──────┘
```

**Why Dueling Architecture?**
- Separates "how good is this state" from "how good is each action"
- Better generalization across actions
- Faster learning in sparse reward settings

#### 4.3 DQN Agent Class
```python
class DQNAgent:
    def __init__(self, ...):
        self.q_network = build_dqn_model(...)      # Main network
        self.target_network = build_dqn_model(...) # Stable target
        self.replay_buffer = ReplayBuffer(...)     # Experience storage
        self.epsilon = 1.0                         # Exploration rate
    
    def select_action(self, state, valid_actions):
        # ε-greedy: random with prob ε, else argmax Q(s,a)
    
    def train(self):
        # Sample batch, compute TD targets, update Q-network
        # Periodically sync target network
```

**Key Hyperparameters**:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 0.001 | Adam optimizer step size |
| Gamma (γ) | 0.95 | Discount factor for future rewards |
| Epsilon Start | 1.0 | Initial exploration (100% random) |
| Epsilon End | 0.05 | Final exploration (5% random) |
| Epsilon Decay | 0.998 | Decay rate per episode |
| Batch Size | 64 | Training batch size |
| Buffer Size | 50,000 | Replay buffer capacity |
| Target Update | Every 50 steps | Sync target network |

---

### **Step 5: Training Loop (Cells 15-17)**

**Training Process**:
```
For each episode (1 to 300):
    1. Reset environment (new simulated user)
    2. For each step (1 to 10):
        a. Agent selects movie (ε-greedy)
        b. Environment returns feedback & reward
        c. Store transition in replay buffer
        d. Sample batch and train Q-network
        e. Decay epsilon
        f. Periodically update target network
    3. Log metrics
```

**What the agent learns**:
- Which movies tend to get positive feedback
- How to adapt to user preferences within a session
- Balance between recommending "safe" popular movies vs exploring

**Training Visualization**:
- Reward curves (should increase over episodes)
- Epsilon decay (exploration decreases)
- Loss curves (should generally decrease)

---

### **Step 6: Evaluation Metrics (Cells 18-19)**

**Metrics Implemented**:

| Metric | What it measures | Formula |
|--------|-----------------|---------|
| **Cumulative Reward** | Total session satisfaction | Σ rewards |
| **Precision@K** | Relevant items in top-K | hits / K |
| **NDCG@K** | Ranking quality | DCG / IDCG |
| **Genre Coverage** | Variety of genres | unique_genres / total_genres |
| **Intra-list Diversity** | How different recommendations are | avg pairwise distance |

**Baseline Comparison**:
- Compare DQN agent vs Random agent
- Shows improvement percentage on each metric

---

### **Step 7: Human Interaction Mode (Cells 20-21)**

**`HumanInteractionSession` Class**:
```python
session = HumanInteractionSession(agent, movie_catalog, processor, session_length=10)
session.run_interactive_session()  # Real human provides feedback
session.run_demo_session(['like', 'skip', 'watch', ...])  # Simulated feedback
```

**Interactive Flow**:
1. Agent recommends a movie
2. Display: title, genre, rating, description
3. User inputs: like/watch/click/skip/ignore
4. System updates user profile
5. Repeat for session length
6. Show session summary

---

### **Step 8: Final Recommender (Cells 22-24)**

**`MovieRecommender` Class**:
```python
recommender = MovieRecommender(agent, movie_catalog, processor)

# Cold-start recommendations
recs = recommender.get_top_k_recommendations(k=10)

# Genre-based recommendations
recs = recommender.recommend_by_genre_preference(['Horror', 'Thriller'], k=10)
```

**How it works**:
1. Takes user state (or generates random for cold-start)
2. Passes through Q-network to get Q-values for all movies
3. Returns top-K movies by Q-value

---

### **Step 9: Model Persistence (Cells 25-26)**

**Saved Files**:
```
models/
├── dqn_recommender_q_network.keras      # Main neural network
├── dqn_recommender_target_network.keras # Target network
├── dqn_recommender_params.pkl           # Hyperparameters
├── feature_processor.pkl                # TF-IDF, SVD, scalers
├── movie_catalog.pkl                    # Movie data + features
└── training_history.pkl                 # Training logs
```

**Loading for Demo**:
```python
agent = DQNAgent.load('models/dqn_recommender')
processor = MovieFeatureProcessor.load('models/feature_processor.pkl')
```

---

## 🚀 How to Run

### First Time (Full Training)
```bash
# 1. Install dependencies
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn

# 2. Open Jupyter notebook
jupyter notebook main.ipynb

# 3. Run all cells in order (1 → 29)
# Training takes ~10-15 minutes for 300 episodes
```

### Demo Mode (Using Pre-trained Model)
```bash
# 1. Open notebook
jupyter notebook main.ipynb

# 2. Run only these cells:
#    - Cell 1 (imports)
#    - Cell 6 (MovieFeatureProcessor class)
#    - Cell 11 (ReplayBuffer class)
#    - Cell 12 (build_dqn_model function)
#    - Cell 13 (DQNAgent class)
#    - Cell 22 (MovieRecommender class)
#    - Cell 26 (load and demo)
```

---

## 🎮 Model Usage

### Get Recommendations for New User
```python
recommender = MovieRecommender(agent, movie_catalog, processor)
recs = recommender.get_top_k_recommendations(k=5)
recommender.display_recommendations(recs)
```

### Get Genre-Specific Recommendations
```python
recs = recommender.recommend_by_genre_preference(
    preferred_genres=['Action', 'Sci-Fi', 'Adventure'],
    k=10
)
```

### Interactive Session with Real Feedback
```python
session = HumanInteractionSession(agent, movie_catalog, processor, session_length=10)
session.run_interactive_session()
# You'll be prompted to give feedback for each recommendation
```

---

## 📈 Evaluation Metrics

### Expected Results (After 300 Episodes)

| Metric | DQN Agent | Random Agent | Improvement |
|--------|-----------|--------------|-------------|
| Cumulative Reward | ~2.5 | ~0.5 | +400% |
| Precision@5 | ~0.15 | ~0.05 | +200% |
| NDCG@5 | ~0.20 | ~0.08 | +150% |
| Diversity | ~0.85 | ~0.90 | Similar |
| Genre Coverage | ~0.35 | ~0.40 | Similar |

**Note**: DQN focuses on relevance (reward), which may slightly reduce diversity.

---

## 📁 File Structure

```
RL_projects/
├── main.ipynb                    # Main Jupyter notebook
├── imdb_movies_2025_cleaned.csv  # Dataset
├── README.md                     # This file
├── training_progress.png         # Learning curves (generated)
├── agent_comparison.png          # Evaluation plots (generated)
└── models/
    ├── dqn_recommender_q_network.keras
    ├── dqn_recommender_target_network.keras
    ├── dqn_recommender_params.pkl
    ├── feature_processor.pkl
    ├── movie_catalog.pkl
    └── training_history.pkl
```

---

## 🔧 Technical Details

### Why These Design Choices?

| Choice | Reason |
|--------|--------|
| **DQN over Policy Gradient** | Discrete action space (movie selection), off-policy learning |
| **Dueling Architecture** | Better value estimation, faster convergence |
| **Experience Replay** | Breaks correlation, sample efficiency |
| **Target Network** | Stabilizes training, prevents oscillation |
| **TF-IDF + SVD** | Efficient text embedding without heavy transformers |
| **ε-greedy** | Simple, effective exploration strategy |

### Limitations & Future Work

1. **Scalability**: Q-network outputs all movie scores → slow for millions of items
   - Solution: Use actor-critic or approximate methods

2. **Exploration**: ε-greedy is simple but not optimal
   - Solution: UCB, Thompson Sampling, or curiosity-driven exploration

3. **User Modeling**: Single preference vector is simplistic
   - Solution: Recurrent networks (LSTM/GRU) for session history

4. **Evaluation**: Simulated users may not reflect real behavior
   - Solution: A/B testing with real users

---

## 📚 References

- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Dueling Network Architectures for Deep RL (Wang et al., 2016)](https://arxiv.org/abs/1511.06581)
- [Deep Reinforcement Learning for Recommender Systems (Survey)](https://arxiv.org/abs/2101.06286)

---

## 👤 Author

**Project**: RL-Based Movie Recommendation System  
**Framework**: TensorFlow/Keras  
**Dataset**: IMDB Movies 2025

---

## 📄 License

This project is for educational and portfolio purposes.

---

*Last Updated: February 2026*
