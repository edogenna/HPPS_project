#ifndef _GSHARE_H_
#define _GSHARE_H_

#include <vector>
#include <stdint.h>
#include <map>      // Necessario per std::map
#include <cassert>  // Necessario per assert

// Struttura per salvare lo stato al momento della predizione
struct SampleHist {
    uint64_t ghist;
    bool tage_pred; // Mantenuto per compatibilità con l'interfaccia
};

class GSHARE {
public:
    // Il costruttore può ancora impostare i parametri di default
    GSHARE(int history_length = 4, int table_size = 1024);

    // Metodi della nuova interfaccia
    void setup();
    void terminate();

    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const;

    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred);
    
    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC);
    
    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC);

private:
    // Membri del predittore GSHARE
    std::vector<int> table;  // Pattern History Table (PHT) con contatori a 2 bit
    int history_length;
    int table_size;

    // Strutture dati per la nuova interfaccia
    SampleHist active_hist; // Cronologia globale "live"
    std::map<uint64_t, SampleHist> pred_time_histories; // Mappa delle cronologie salvate

    // Metodi helper interni
    int get_index(uint64_t pc, uint64_t ghr) const;
    void update_predictor_table(uint64_t pc, bool taken, const SampleHist& hist_to_use);

    // Metodo di aggiornamento chiamato dalla funzione pubblica update
    void update(uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC, const SampleHist& hist_to_use);
};

extern GSHARE cond_predictor_impl; // Dichiarazione dell'istanza globale del predittore GSHARE

#endif