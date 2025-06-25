#ifndef _TOURNAMENT_PREDICTOR_H_
#define _TOURNAMENT_PREDICTOR_H_

#include <vector>
#include <cstdint>

// ============================================================================
// ==                CONFIGURAZIONE DEL TOURNAMENT PREDICTOR                 ==
// ============================================================================
#define LOG_LOCAL_PREDICTOR_SIZE 14
#define LOG_GLOBAL_PREDICTOR_SIZE 14
#define LOG_CHOOSER_SIZE 14
#define GLOBAL_HISTORY_LENGTH 12
// ============================================================================

class TOURNAMENT_PREDICTOR {
private:
    // --- Componenti Hardware del Predittore ---
    std::vector<int8_t> local_predictor_table;
    std::vector<int8_t> global_predictor_table;
    std::vector<int8_t> chooser_table;
    uint64_t ghr; // Global History Register

    // --- Funzioni Helper Private ---
    void saturating_increment(int8_t &counter);
    void saturating_decrement(int8_t &counter);

public:
    // --- Interfaccia Pubblica per il Simulatore CBP ---

    // Costruttore: dichiara la funzione di inizializzazione
    TOURNAMENT_PREDICTOR();

    // Funzione principale di predizione
    bool get_cond_dir_prediction(uint64_t pc);

    // Funzione principale di aggiornamento
    void update_predictor(uint64_t pc, bool taken, bool pred);
};

extern TOURNAMENT_PREDICTOR cond_predictor_impl; // Dichiarazione dell'istanza globale del predittore Tournament

#endif // _TOURNAMENT_PREDICTOR_H_