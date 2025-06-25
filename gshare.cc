#include "my_cond_branch_predictor.h"
#include <algorithm> // Necessario per std::fill

GSHARE cond_predictor_impl(12, 4096);

// Il costruttore ora salva solo i parametri
GSHARE::GSHARE(int history_length, int table_size)
    : history_length(history_length), table_size(table_size) {
    // L'inizializzazione vera e propria avviene in setup()
    active_hist.ghist = 0;
    active_hist.tage_pred = false;
}

// Inizializza le strutture dati del predittore
void GSHARE::setup() {
    table.resize(table_size, 0); // Inizializza i contatori a "strongly not taken"
}

// Funzione di cleanup (se necessaria)
void GSHARE::terminate() {
    // In questo caso non c'è nulla da fare
}

// Funzione di esempio per ottenere un ID univoco per l'istruzione
uint64_t GSHARE::get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
    assert(piece < 16);
    return (seq_no << 4) | (piece & 0x000F);
}

// Calcola l'indice per la tabella PHT usando la cronologia fornita
int GSHARE::get_index(uint64_t pc, uint64_t ghr) const {
    // L'indice è un XOR tra il PC (o parte di esso) e la cronologia globale
    uint64_t pc_masked = pc & (table_size - 1);
    return (ghr ^ pc_masked) % table_size;
}

// Esegue la predizione
bool GSHARE::predict(uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred) {
    // Salva lo stato corrente della cronologia per un uso futuro nell'aggiornamento
    active_hist.tage_pred = tage_pred;
    const uint64_t inst_id = get_unique_inst_id(seq_no, piece);
    pred_time_histories.emplace(inst_id, active_hist);

    // Esegue la predizione GSHARE usando la cronologia ATTIVA
    int index = get_index(PC, active_hist.ghist);
    int state = table[index];

    // La predizione è "taken" se il contatore è in uno stato di "weakly" o "strongly taken"
    return state >= 2;
}

// Aggiorna la cronologia globale "live"
void GSHARE::history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
    // Sposta a sinistra la cronologia e inserisce il nuovo risultato del branch
    active_hist.ghist = active_hist.ghist << 1;
    if (taken) {
        active_hist.ghist |= 1;
    }
    // Mantiene la cronologia entro la sua lunghezza massima
    active_hist.ghist &= (1ULL << history_length) - 1;
}

// Funzione di aggiornamento principale (interfaccia pubblica)
void GSHARE::update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
    const auto pred_hist_key = get_unique_inst_id(seq_no, piece);
    const auto& pred_time_history = pred_time_histories.at(pred_hist_key);

    // Chiama la funzione di aggiornamento interna con la cronologia salvata
    update(PC, resolveDir, predDir, nextPC, pred_time_history);

    // Rimuove la voce dalla mappa poiché non è più necessaria
    pred_time_histories.erase(pred_hist_key);
}

// Funzione di aggiornamento interna che contiene la logica GSHARE
void GSHARE::update(uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC, const SampleHist& hist_to_use) {
    // Calcola l'indice usando la cronologia salvata al momento della predizione
    int index = get_index(PC, hist_to_use.ghist);
    int& state = table[index];

    // Aggiorna il contatore a 2 bit
    if (resolveDir) { // Il branch è stato preso (Taken)
        if (state < 3) state++;
    } else { // Il branch non è stato preso (Not Taken)
        if (state > 0) state--;
    }
}