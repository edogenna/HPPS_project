#include "my_cond_branch_predictor.h"

BHTPredictor cond_predictor_impl(1024);  

BHTPredictor::BHTPredictor(int table_size) {
    table.resize(table_size, 0);  
}

int BHTPredictor::get_index(uint64_t address) const {
    return address % table.size();  
}

bool BHTPredictor::predict(uint64_t seq_no, uint8_t piece, uint64_t pc, bool tage_sc_l_pred) {
    int index = get_index(pc);    
    int state = table[index];     

    return state >= 2;
}

void BHTPredictor::update(uint64_t seq_no, uint8_t piece, uint64_t pc, bool resolve_dir, bool pred_dir, uint64_t next_pc) {
    int index = get_index(pc);   
    int &state = table[index];    
    
    if (resolve_dir) {
        if (state < 3) {
            state++;  
        }
    }
    
    else {
        if (state > 0) {
            state--;  
        }
    }
}

void BHTPredictor::setup() {
}

void BHTPredictor::history_update(uint64_t seq_no, uint8_t piece, uint64_t pc, bool resolve_dir, uint64_t next_pc) {
}

void BHTPredictor::terminate() {
}

