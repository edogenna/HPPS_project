#include "tournament_predictor.h" // Include la dichiarazione della nostra classe

#include <cmath> // Necessario per la logica di inizializzazione

// Implementazione del Costruttore

TOURNAMENT_PREDICTOR cond_predictor_impl; // Dichiarazione dell'istanza globale del predittore Tournament

TOURNAMENT_PREDICTOR::TOURNAMENT_PREDICTOR() {
    // Inizializza le tabelle con le dimensioni definite nel file .h
    local_predictor_table.resize(1 << LOG_LOCAL_PREDICTOR_SIZE);
    global_predictor_table.resize(1 << LOG_GLOBAL_PREDICTOR_SIZE);
    chooser_table.resize(1 << LOG_CHOOSER_SIZE);

    // Imposta tutti i contatori a uno stato iniziale di "debolmente non preso" (1)
    for (auto &entry : local_predictor_table) entry = 1;
    for (auto &entry : global_predictor_table) entry = 1;
    for (auto &entry : chooser_table) entry = 1; // Preferenza iniziale debole per il locale

    ghr = 0; // Azzera la storia globale
}

// Implementazione della funzione di predizione
bool TOURNAMENT_PREDICTOR::get_cond_dir_prediction(uint64_t pc) {
    // 1. Predizione Locale
    uint32_t local_index = pc % (1 << LOG_LOCAL_PREDICTOR_SIZE);
    bool local_prediction = (local_predictor_table[local_index] >= 2);

    // 2. Predizione Globale (GShare)
    uint64_t history_mask = (1 << GLOBAL_HISTORY_LENGTH) - 1;
    uint32_t global_index = (pc ^ (ghr & history_mask)) % (1 << LOG_GLOBAL_PREDICTOR_SIZE);
    bool global_prediction = (global_predictor_table[global_index] >= 2);

    // 3. Decisione del Selettore
    uint32_t chooser_index = pc % (1 << LOG_CHOOSER_SIZE);
    bool use_global_predictor = (chooser_table[chooser_index] >= 2);

    return use_global_predictor ? global_prediction : local_prediction;
}

// Implementazione della funzione di aggiornamento
void TOURNAMENT_PREDICTOR::update_predictor(uint64_t pc, bool taken, bool pred) {
    // Ottieni di nuovo le predizioni per vedere chi aveva ragione
    uint32_t local_index = pc % (1 << LOG_LOCAL_PREDICTOR_SIZE);
    bool local_prediction = (local_predictor_table[local_index] >= 2);
    uint64_t history_mask = (1 << GLOBAL_HISTORY_LENGTH) - 1;
    uint32_t global_index = (pc ^ (ghr & history_mask)) % (1 << LOG_GLOBAL_PREDICTOR_SIZE);
    bool global_prediction = (global_predictor_table[global_index] >= 2);

    bool local_correct = (local_prediction == taken);
    bool global_correct = (global_prediction == taken);

    // Aggiorna il selettore
    uint32_t chooser_index = pc % (1 << LOG_CHOOSER_SIZE);
    if (global_correct && !local_correct) {
        saturating_increment(chooser_table[chooser_index]);
    } else if (!global_correct && local_correct) {
        saturating_decrement(chooser_table[chooser_index]);
    }

    // Aggiorna i contatori dei predittori di base
    if (taken) {
        saturating_increment(local_predictor_table[local_index]);
        saturating_increment(global_predictor_table[global_index]);
    } else {
        saturating_decrement(local_predictor_table[local_index]);
        saturating_decrement(global_predictor_table[global_index]);
    }

    // Aggiorna la storia globale alla fine
    ghr = ((ghr << 1) | taken);
}

// Implementazione delle funzioni helper
void TOURNAMENT_PREDICTOR::saturating_increment(int8_t &counter) {
    if (counter < 3) {
        counter++;
    }
}

void TOURNAMENT_PREDICTOR::saturating_decrement(int8_t &counter) {
    if (counter > 0) {
        counter--;
    }
}