import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from collections import defaultdict
import re
import seaborn as sns
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

class WimbledonPredictor:
    def __init__(self):
        self.elo_ratings = defaultdict(lambda: 1500)
        self.grass_elo_ratings = defaultdict(lambda: 1500)
        self.player_matches = defaultdict(int)
        self.grass_matches = defaultdict(int)
        self.match_history = defaultdict(list)
        self.h2h_records = defaultdict(lambda: defaultdict(list))
        self.player_stats = defaultdict(lambda: {
            'total_matches': 0,
            'grass_matches': 0,
            'recent_form': [],
            'grass_form': [],
            'wimbledon_performance': []
        })

        
    def normalize_player_name(self, name):
        """Normalizza i nomi dei giocatori per matching consistente"""
        if pd.isna(name):
            return ""
        # Rimuove spazi extra e standardizza il formato
        name = str(name).strip()
        # Gestisce il formato "Cognome Iniziale."
        if '.' in name and len(name.split()) == 2:
            parts = name.split()
            if len(parts[1]) == 2 and parts[1].endswith('.'):
                return name
        return name
    
    def calculate_k_factor(self, matches_played, tournament_importance=1.0, round_importance=1.0):
        """Calcola il K-factor basato su esperienza e importanza del match"""
        # K-factor base diminuisce con l'esperienza
        if matches_played < 10:
            base_k = 40
        elif matches_played < 30:
            base_k = 32
        elif matches_played < 50:
            base_k = 24
        else:
            base_k = 16
            
        # Aggiusta per importanza torneo e round
        adjusted_k = base_k * tournament_importance * round_importance
        return max(10, min(50, adjusted_k))  # Limiti ragionevoli
    
    def get_tournament_importance(self, tournament_name, surface):
        """Determina l'importanza del torneo"""
        tournament_lower = tournament_name.lower()
        
        # Grand Slam
        if any(slam in tournament_lower for slam in ['wimbledon', 'us open', 'french open', 'australian open']):
            if 'wimbledon' in tournament_lower:
                return 1.3  # Wimbledon ha peso maggiore
            return 1.2
        
        # Masters 1000
        if any(master in tournament_lower for master in ['indian wells', 'miami', 'monte carlo', 'madrid', 'rome', 'canada', 'cincinnati', 'shanghai', 'paris']):
            return 1.1
            
        # Tornei su erba (peso maggiore per la specializzazione)
        if surface == 'Grass':
            return 1.175
            
        return 1.0
    
    def get_round_importance(self, round_name):
        """Determina l'importanza del round"""
        if pd.isna(round_name):
            return 1.0
            
        round_lower = str(round_name).lower()
        if 'final' in round_lower:
            return 1.05
        elif 'semifinal' in round_lower or 'semi' in round_lower:
            return 1.05
        elif 'quarterfinal' in round_lower or 'quarter' in round_lower:
            return 1.02
        return 1.0
    
    def expected_score(self, rating1, rating2, best_of=3):
        """Calcola la probabilità di vittoria basata sui rating ELO"""
        diff = rating2 - rating1
        expected = 1 / (1 + 10 ** (diff / 400))
        
        # Aggiustamento per best-of-5 (Grand Slam)
        if best_of == 5:
            # Il favorito ha più probabilità nei match lunghi
            if expected > 0.5:
                expected = expected + (expected - 0.5) * 0.15
            else:
                expected = expected - (0.5 - expected) * 0.15
                
        return max(0.01, min(0.99, expected))
    
    def update_elo(self, winner_elo, loser_elo, winner_matches, loser_matches, 
                   tournament_importance, round_importance, score):
        """Aggiorna i rating ELO dopo un match"""
        # Calcola K-factors
        winner_k = self.calculate_k_factor(winner_matches, tournament_importance, round_importance)
        loser_k = self.calculate_k_factor(loser_matches, tournament_importance, round_importance)
        
        # Probabilità attese
        winner_expected = self.expected_score(winner_elo, loser_elo)
        loser_expected = 1 - winner_expected
        
        # Bonus per vittorie nette vs vittorie sofferte
        score_bonus = self.calculate_score_bonus(score)
        
        # Aggiornamenti
        winner_new = winner_elo + winner_k * (1 - winner_expected) * score_bonus
        loser_new = loser_elo + loser_k * (0 - loser_expected)
        
        return winner_new, loser_new
    
    def calculate_score_bonus(self, score):
        """Calcola bonus basato su quanto netta è stata la vittoria"""
        if pd.isna(score) or score == "":
            return 1.0
            
        try:
            # Conta i set vinti
            sets = str(score).split()
            winner_sets = 0
            loser_sets = 0
            
            for set_score in sets:
                if '-' in set_score:
                    games = set_score.split('-')
                    if len(games) == 2:
                        if int(games[0]) > int(games[1]):
                            winner_sets += 1
                        else:
                            loser_sets += 1
            
            # Bonus per vittorie nette
            if winner_sets >= 3 and loser_sets == 0:  # 3-0 o 4-0
                return 1.15
            elif winner_sets >= 3 and loser_sets == 1:  # 3-1 o 4-1
                return 1.1
            else:
                return 1.0
                
        except:
            return 1.0
    
    def apply_time_decay(self, date_str, base_weight=1.0):
        """Applica decay temporale ai risultati"""
        try:
            match_date = pd.to_datetime(date_str)
            cutoff_2024 = pd.to_datetime('2024-01-01')
            
            if match_date >= cutoff_2024:
                return base_weight * 0.55  # Peso maggiore ai dati recenti
            else:
                return base_weight * 0.45  # Peso minore ai dati più vecchi
        except:
            return base_weight * 0.45
    
    def initialize_player_elos_intelligently(self, df):
        """Inizializza gli ELO in modo più intelligente basandosi sui primi risultati"""
        print("Inizializzazione intelligente degli ELO...")
        
        # Prima passata: identifica i top player dai primi risultati del 2021
        early_2021 = df[df['Date'] < '2021-04-01'].copy()
        player_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'opponents': []})
        
        for _, row in early_2021.iterrows():
            winner = self.normalize_player_name(row['Winner'])
            loser = self.normalize_player_name(row['Player_1']) if row['Winner'] == row['Player_2'] else self.normalize_player_name(row['Player_2'])
            
            if winner and loser:
                player_performance[winner]['wins'] += 1
                player_performance[winner]['opponents'].append(loser)
                player_performance[loser]['losses'] += 1
                player_performance[loser]['opponents'].append(winner)
        
        # Calcola ELO iniziali aggiustati
        for player, stats in player_performance.items():
            total_matches = stats['wins'] + stats['losses']
            if total_matches >= 3:  # Solo per giocatori con abbastanza dati
                win_rate = stats['wins'] / total_matches
                
                # Aggiustamento basato su win rate e qualità degli avversari
                if win_rate > 0.75:
                    self.elo_ratings[player] = 1650  # Top performers
                    self.grass_elo_ratings[player] = 1650
                elif win_rate > 0.6:
                    self.elo_ratings[player] = 1580
                    self.grass_elo_ratings[player] = 1580
                elif win_rate > 0.4:
                    self.elo_ratings[player] = 1520
                    self.grass_elo_ratings[player] = 1520
                else:
                    self.elo_ratings[player] = 1450  # Performers più deboli
                    self.grass_elo_ratings[player] = 1450
    def load_and_process_data(self, file_path):
        """Carica e processa i dati ATP dal 2021"""
        print("Caricamento dati ATP...")
        df = pd.read_csv(file_path, on_bad_lines='skip')
        
        # Filtra dal 2021 fino al 1 luglio 2025
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= '2021-01-01') & (df['Date'] <= '2025-07-01')].copy()
        
        # Ordina per data per processare cronologicamente
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Processando {len(df)} partite dal 2021 al 1 luglio 2025...")
        
        # Ordina per data per processare cronologicamente
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Processando {len(df)} partite dal 2021...")
        
        # Inizializzazione intelligente degli ELO
        self.initialize_player_elos_intelligently(df)
        
        # Processa ogni partita per costruire ELO e statistiche
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processate {idx} partite...")
                
            self.process_match(row)
        
        print("Processamento dati completato!")
        return df
    
    def process_match(self, row):
        """Processa una singola partita per aggiornare ELO e statistiche"""
        player1 = self.normalize_player_name(row['Player_1'])
        player2 = self.normalize_player_name(row['Player_2'])
        winner = self.normalize_player_name(row['Winner'])
        surface = row['Surface']
        tournament = row['Tournament']
        round_name = row['Round']
        score = row['Score']
        date = row['Date']
        
        if not player1 or not player2 or not winner:
            return
        
        # Determina vincitore e perdente
        if winner == player1:
            winner_name, loser_name = player1, player2
        else:
            winner_name, loser_name = player2, player1
        
        # Ottieni ELO attuali
        winner_elo = self.elo_ratings[winner_name]
        loser_elo = self.elo_ratings[loser_name]
        winner_grass_elo = self.grass_elo_ratings[winner_name]
        loser_grass_elo = self.grass_elo_ratings[loser_name]
        
        # Conta partite giocate
        winner_matches = self.player_matches[winner_name]
        loser_matches = self.player_matches[loser_name]
        
        # Ottieni importanza torneo e round
        tournament_importance = self.get_tournament_importance(tournament, surface)
        round_importance = self.get_round_importance(round_name)
        
        # Aggiorna ELO generali
        new_winner_elo, new_loser_elo = self.update_elo(
            winner_elo, loser_elo, winner_matches, loser_matches,
            tournament_importance, round_importance, score
        )
        
        # Applica decay temporale
        time_weight = self.apply_time_decay(date)
        
        # Aggiorna ELO con peso temporale
        self.elo_ratings[winner_name] = winner_elo + (new_winner_elo - winner_elo) * time_weight
        self.elo_ratings[loser_name] = loser_elo + (new_loser_elo - loser_elo) * time_weight
        
        # Aggiorna ELO su erba se necessario
        if surface == 'Grass':
            new_winner_grass_elo, new_loser_grass_elo = self.update_elo(
                winner_grass_elo, loser_grass_elo, 
                self.grass_matches[winner_name], self.grass_matches[loser_name],
                tournament_importance, round_importance, score
            )
            
            self.grass_elo_ratings[winner_name] = winner_grass_elo + (new_winner_grass_elo - winner_grass_elo) * time_weight
            self.grass_elo_ratings[loser_name] = loser_grass_elo + (new_loser_grass_elo - loser_grass_elo) * time_weight
            
            self.grass_matches[winner_name] += 1
            self.grass_matches[loser_name] += 1
        
        # Aggiorna contatori partite
        self.player_matches[winner_name] += 1
        self.player_matches[loser_name] += 1
        
        # Aggiorna H2H
        self.h2h_records[winner_name][loser_name].append({
            'date': date, 'surface': surface, 'tournament': tournament, 'result': 'W'
        })
        self.h2h_records[loser_name][winner_name].append({
            'date': date, 'surface': surface, 'tournament': tournament, 'result': 'L'
        })
        
        # Aggiorna statistiche giocatori
        for player in [winner_name, loser_name]:
            result = 'W' if player == winner_name else 'L'
            match_info = {
                'date': date, 'surface': surface, 'tournament': tournament, 
                'result': result, 'opponent': loser_name if player == winner_name else winner_name
            }
            
            self.player_stats[player]['total_matches'] += 1
            self.player_stats[player]['recent_form'].append(match_info)
            
            if surface == 'Grass':
                self.player_stats[player]['grass_matches'] += 1
                self.player_stats[player]['grass_form'].append(match_info)
                
            if 'wimbledon' in tournament.lower():
                self.player_stats[player]['wimbledon_performance'].append(match_info)
    
    def calculate_h2h_weight(self, player1, player2):
        """Calcola il peso H2H tra due giocatori"""
        if player2 not in self.h2h_records[player1]:
            return 0.5, 0  # Nessun precedente
        
        matches = self.h2h_records[player1][player2]
        if not matches:
            return 0.5, 0
        
        recent_wins = 0
        old_wins = 0
        recent_total = 0
        old_total = 0
        
        cutoff_2024 = pd.to_datetime('2024-01-01')
        
        for match in matches:
            match_date = pd.to_datetime(match['date'])
            is_recent = match_date >= cutoff_2024
            is_grass = match['surface'] == 'Grass'
            
            if is_recent:
                recent_total += 1
                if match['result'] == 'W':
                    # Peso maggiore se su erba
                    weight = 0.7 if not is_grass else 0.6
                    recent_wins += weight
            else:
                old_total += 1
                if match['result'] == 'W':
                    weight = 0.3 if not is_grass else 0.4
                    old_wins += weight
        
        total_matches = recent_total + old_total
        if total_matches == 0:
            return 0.5, 0
            
        h2h_score = (recent_wins + old_wins) / total_matches
        return h2h_score, total_matches
    
    def calculate_recent_form(self, player, months=12):
        """Calcola la forma recente di un giocatore"""
        if player not in self.player_stats:
            return 0.5
        
        cutoff_date = datetime.now() - timedelta(days=months*30)
        recent_matches = []
        
        for match in self.player_stats[player]['recent_form']:
            if pd.to_datetime(match['date']) >= cutoff_date:
                recent_matches.append(match)
        
        # Prendi le ultime 15 partite o quelle disponibili
        recent_matches = sorted(recent_matches, key=lambda x: x['date'], reverse=True)[:15]
        
        if not recent_matches:
            return 0.5
        
        wins = sum(1 for match in recent_matches if match['result'] == 'W')
        total = len(recent_matches)
        
        # Applica decay temporale alle partite più vecchie
        weighted_wins = 0
        weighted_total = 0
        
        for i, match in enumerate(recent_matches):
            weight = 1.0 - (i * 0.05)  # Decay lineare
            weight = max(0.5, weight)  # Peso minimo
            
            # Bonus per partite su erba
            if match['surface'] == 'Grass':
                weight *= 1.2
                
            weighted_total += weight
            if match['result'] == 'W':
                weighted_wins += weight
        
        return weighted_wins / weighted_total if weighted_total > 0 else 0.5
    
    def get_combined_elo(self, player):
        """Calcola ELO combinato (70% generale + 30% erba)"""
        general_elo = self.elo_ratings[player]
        grass_elo = self.grass_elo_ratings[player]
        
        # Se il giocatore ha poche partite su erba, usa più peso al generale
        grass_matches = self.grass_matches[player]
        if grass_matches < 5:
            grass_weight = min(0.3, grass_matches * 0.06)  # Peso cresce gradualmente
            general_weight = 1 - grass_weight
        else:
            general_weight = 0.7
            grass_weight = 0.3
        
        combined_elo = general_elo * general_weight + grass_elo * grass_weight
        return combined_elo
    
    def create_features(self, player1, player2):
        """Crea le features per la predizione"""
        # ELO features
        elo1 = self.get_combined_elo(player1)
        elo2 = self.get_combined_elo(player2)
        elo_diff = elo1 - elo2
        elo1_grass = self.grass_elo_ratings[player1]
        elo2_grass = self.grass_elo_ratings[player2]
        elo_grass_diff = elo1_grass - elo2_grass
        elo_gap_ratio = elo_diff / max(1, elo1 + elo2)

        # H2H features
        h2h_score, h2h_matches = self.calculate_h2h_weight(player1, player2)
        
        # Form features
        form1 = self.calculate_recent_form(player1)
        form2 = self.calculate_recent_form(player2)
        form_diff = form1 - form2
        
        # Experience features
        total_matches1 = self.player_matches[player1]
        total_matches2 = self.player_matches[player2]
        grass_matches1 = self.grass_matches[player1]
        grass_matches2 = self.grass_matches[player2]
        
        # Grass experience ratio
        grass_exp1 = grass_matches1 / max(1, total_matches1)
        grass_exp2 = grass_matches2 / max(1, total_matches2)
        grass_exp_diff = grass_exp1 - grass_exp2
        
        # Wimbledon experience
        wimbledon_matches1 = len(self.player_stats[player1]['wimbledon_performance'])
        wimbledon_matches2 = len(self.player_stats[player2]['wimbledon_performance'])
        
        features = [
        elo_diff,
       elo_grass_diff,
       elo_gap_ratio,
       h2h_score - 0.5,
       form_diff,
       np.log1p(total_matches1) - np.log1p(total_matches2),
       grass_exp_diff,
       np.log1p(wimbledon_matches1) - np.log1p(wimbledon_matches2),
       h2h_matches
       ]

        return features
    
    def prepare_training_data(self, df):
        """Prepara i dati di training"""
        print("Preparazione dati di training...")

        X = []
        y = []

        # Filtra partite su erba per training
        grass_matches = df[df['Surface'] == 'Grass'].copy()

        for _, row in grass_matches.iterrows():
            player1 = self.normalize_player_name(row['Player_1'])
            player2 = self.normalize_player_name(row['Player_2'])
            winner = self.normalize_player_name(row['Winner'])

            if not player1 or not player2 or not winner:
                continue

            # Soglia più bassa per training data
            if (self.player_matches[player1] >= 2 and self.player_matches[player2] >= 2):
                features = self.create_features(player1, player2)
                X.append(features)
                y.append(1 if winner == player1 else 0)

        return np.array(X), np.array(y)

    def train_models(self, X, y):
        """Addestra diversi modelli e seleziona il migliore"""
        print("Training modelli...")

        # Split dei dati
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.9,
                gamma=0.2,
                min_child_weight=2,
                reg_alpha=0.5,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_split=8,
                min_samples_leaf=3,
                random_state=42,
                class_weight='balanced'
            ),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            )
        }

        best_model = None
        best_score = 0
        results = {}

        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_pred)

            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'model': model
            }

            print(f"{name}:")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Test Accuracy: {test_accuracy:.4f}")

            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model = model

        self.best_model = best_model
        return results

    def predict_match(self, player1, player2):
        """Predice il risultato di una partita"""
        # Normalizza nomi
        player1 = self.normalize_player_name(player1)
        player2 = self.normalize_player_name(player2)

        # Controlla se i giocatori hanno abbastanza dati
        player1_matches = self.player_matches[player1]
        player2_matches = self.player_matches[player2]

        # Soglia più flessibile: almeno 3 partite totali
        if (player1_matches < 3 or player2_matches < 3):
            # Fallback su probabilità ELO pura con aggiustamenti
            elo1 = self.get_combined_elo(player1)
            elo2 = self.get_combined_elo(player2)

            # Aggiustamento per giocatori con pochi dati
            if player1_matches < 3:
                elo1 *= 0.95  # Penalty leggera per poca esperienza
            if player2_matches < 3:
                elo2 *= 0.95

            prob = self.expected_score(elo1, elo2, best_of=5)
            return prob, f"ELO-based (P1:{player1_matches} matches, P2:{player2_matches} matches)"

        # Usa ML se entrambi hanno almeno 3 partite, altrimenti sistema ibrido
        if player1_matches >= 3 and player2_matches >= 3:
            # Crea features
            features = self.create_features(player1, player2)
            features_array = np.array(features).reshape(1, -1)

            # Predizione del modello
            if hasattr(self.best_model, 'predict_proba'):
                prob = self.best_model.predict_proba(features_array)[0][1]
            else:
                prob = self.best_model.predict(features_array)[0]

            # Combina con ELO per maggiore robustezza
            elo1 = self.get_combined_elo(player1)
            elo2 = self.get_combined_elo(player2)
            elo_prob = self.expected_score(elo1, elo2, best_of=5)

            # Peso variabile in base all'esperienza
            ml_weight = min(0.7, (player1_matches + player2_matches) / 20)
            elo_weight = 1 - ml_weight

            final_prob = ml_weight * prob + elo_weight * elo_prob

            return final_prob, f"ML Model ({ml_weight:.1%}) + ELO ({elo_weight:.1%})"

        # Sistema ibrido per giocatori con poche partite
        else:
            elo1 = self.get_combined_elo(player1)
            elo2 = self.get_combined_elo(player2)
            elo_prob = self.expected_score(elo1, elo2, best_of=5)

            # Aggiusta in base alla forma recente se disponibile
            form1 = self.calculate_recent_form(player1)
            form2 = self.calculate_recent_form(player2)

            if form1 != 0.5 or form2 != 0.5:  # Se abbiamo dati di forma
                form_factor = (form1 - form2) * 0.1  # Peso leggero alla forma
                adjusted_prob = elo_prob + form_factor
                adjusted_prob = max(0.1, min(0.9, adjusted_prob))  # Limiti
                return adjusted_prob, f"ELO + Form (P1:{player1_matches}, P2:{player2_matches} matches)"

            return elo_prob, f"ELO-based (P1:{player1_matches}, P2:{player2_matches} matches)"

    def load_main_draw(self, file_path):
        """Carica il main draw di Wimbledon"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Errore nel caricamento del main draw: {e}")
            return None

    def predict_tournament(self, main_draw_path):
        """Predice tutti i match del torneo"""
        draw_df = self.load_main_draw(main_draw_path)
        if draw_df is None:
            return

        print("\n=== PREDICTIONS WIMBLEDON 2025 ===\n")

        predictions = []

        for _, row in draw_df.iterrows():
            player1 = row['Player_1']
            player2 = row['Player_2']
            round_name = row['Round']

            prob, method = self.predict_match(player1, player2)

            # Determina il favorito
            if prob > 0.5:
                favorite = player1
                confidence = prob
            else:
                favorite = player2
                confidence = 1 - prob

            predictions.append({
                'Player_1': player1,
                'Player_2': player2,
                'Round': round_name,
                'Predicted_Winner': favorite,
                'Confidence': confidence,
                'Method': method
            })

            print(f"{round_name}: {player1} vs {player2}")
            print(f"  Prediction: {favorite} ({confidence:.1%})")
            print(f"  Method: {method}")

            # Mostra dettagli per match interessanti
            if abs(prob - 0.5) < 0.1:  # Match molto equilibrati
                elo1 = self.get_combined_elo(player1)
                elo2 = self.get_combined_elo(player2)
                print(f"  ELO: {player1} ({elo1:.0f}) vs {player2} ({elo2:.0f})")
            print()

        return predictions

    def get_player_profile(self, player_name):
        """Ottieni profilo dettagliato di un giocatore"""
        player = self.normalize_player_name(player_name)

        if player not in self.elo_ratings:
            return f"Giocatore {player_name} non trovato nel database"

        profile = {
            'name': player,
            'total_elo': self.elo_ratings[player],
            'grass_elo': self.grass_elo_ratings[player],
            'combined_elo': self.get_combined_elo(player),
            'total_matches': self.player_matches[player],
            'grass_matches': self.grass_matches[player],
            'recent_form': self.calculate_recent_form(player),
            'wimbledon_matches': len(self.player_stats[player]['wimbledon_performance'])
        }

        return profile

def main():
    # Inizializza il predictor
    predictor = WimbledonPredictor()
    
    # Carica e processa i dati
    df = predictor.load_and_process_data('atp_tennis.csv')
    
    # Prepara i dati di training
    X, y = predictor.prepare_training_data(df)
    print(f"Dataset di training: {len(X)} partite su erba")

    # --- INIZIO CODICE PER pairplot ---
    columns = [
        'elo_diff',
        'elo_grass_diff',
        'elo_gap_ratio',
        'h2h_score_diff',
        'form_diff',
        'log_total_matches_diff',
        'grass_exp_diff',
        'log_wimbledon_matches_diff',
        'h2h_matches'
    ]

    df_features = pd.DataFrame(X, columns=columns)
    df_features['winner_player1'] = y

    sns.pairplot(df_features, hue='winner_player1', corner=True, plot_kws={'alpha':0.6, 's':40})
    plt.show()
    # --- FINE CODICE PER pairplot ---
    
    # Addestra i modelli
    results = predictor.train_models(X, y)
    
    # Fai le predizioni per il torneo
    predictions = predictor.predict_tournament('wimbledon_main_draw.csv')
    
    # Salva le predizioni
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv('wimbledon_2025_predictions.csv', index=False)
        print("Predizioni salvate in 'wimbledon_2025_predictions.csv'")
    
    return predictor

if __name__ == "__main__":
    predictor = main()


