#ifndef _BHT_PREDICTOR_H_
#define _BHT_PREDICTOR_H_

#include <vector>
#include <stdint.h>
#include <assert.h>

class BHTPredictor {
public:
    BHTPredictor(int table_size);
    
    bool predict(uint64_t seq_no, uint8_t piece, uint64_t pc, bool tage_sc_l_pred);
    
    void update(uint64_t seq_no, uint8_t piece, uint64_t pc, bool resolve_dir, bool pred_dir, uint64_t next_pc);
    
    void setup();
    
    void history_update(uint64_t seq_no, uint8_t piece, uint64_t pc, bool resolve_dir, uint64_t next_pc);
    
    void terminate();

private:
    std::vector<int> table;  
    
    int get_index(uint64_t address) const;
};

extern BHTPredictor cond_predictor_impl;  

#endif 