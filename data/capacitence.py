import numpy as np
from enum import Enum

class CAP(Enum):
    ROW_SHIFT = "ROW"
    COLUMN_SHIFT = "COL"
    DIAGONAL_SHIFT_1 = "DIAG_1"
    DIAGONAL_SHIFT_2 = "DIAG_2"


class CAPMEAS():
    def __init__(self, row_shift=[], column_shift=[], diag_shift_1=[], diag_shift_2=[]):
        self.row_shift = row_shift
        self.column_shift = column_shift
        self.diag_shift_1 = diag_shift_1
        self.diag_shift_2 = diag_shift_2
    
    
    def get_all(self):
        return self.row_shift, self.column_shift, self.diag_shift_1, self.diag_shift_2
    
    def merge_all(self):
        cap_all = np.concatenate((self.row_shift, self.column_shift, self.diag_shift_1, self.diag_shift_2), axis=2)
        return cap_all
    
    def get(self, type):
        if type == CAP.ROW_SHIFT or type == CAP.ROW_SHIFT.value: 
            return self.row_shift 
        if type == CAP.COLUMN_SHIFT or type == CAP.COLUMN_SHIFT.value: 
            return self.column_shift 
        if type == CAP.DIAGONAL_SHIFT_1 or type == CAP.DIAGONAL_SHIFT_1.value: 
            return self.diag_shift_1
        if type == CAP.DIAGONAL_SHIFT_2 or type == CAP.DIAGONAL_SHIFT_2.value: 
            return self.diag_shift_2

    @property
    def row_shift_stats(self): 
        return np.mean(self.row_shift), np.std(self.row_shift), np.min(self.row_shift), np.max(self.row_shift)

    @property
    def column_shift_stats(self): 
        return np.mean(self.column_shift), np.std(self.column_shift), np.min(self.column_shift), np.max(self.column_shift)

    @property
    def diag_shift_1_stats(self): 
        return np.mean(self.diag_shift_1), np.std(self.diag_shift_1), np.min(self.diag_shift_1), np.max(self.diag_shift_1)

    @property
    def diag_shift_2_stats(self): 
        return np.mean(self.diag_shift_2), np.std(self.diag_shift_2), np.min(self.diag_shift_2), np.max(self.diag_shift_2)

    @property
    def mean(self):
        return {CAP.ROW_SHIFT.value: np.mean(self.row_shift), 
                CAP.COLUMN_SHIFT.value: np.mean(self.column_shift), 
                CAP.DIAGONAL_SHIFT_1.value: np.mean(self.diag_shift_1), 
                CAP.DIAGONAL_SHIFT_2.value: np.mean(self.diag_shift_2)}

    @property
    def std(self):
        return {CAP.ROW_SHIFT.value: np.std(self.row_shift), 
                CAP.COLUMN_SHIFT.value: np.std(self.column_shift), 
                CAP.DIAGONAL_SHIFT_1.value: np.std(self.diag_shift_1), 
                CAP.DIAGONAL_SHIFT_2.value: np.std(self.diag_shift_2)}

    @property
    def min(self):
        return {CAP.ROW_SHIFT.value: np.min(self.row_shift), 
                CAP.COLUMN_SHIFT.value: np.min(self.column_shift), 
                CAP.DIAGONAL_SHIFT_1.value: np.min(self.diag_shift_1), 
                CAP.DIAGONAL_SHIFT_2.value: np.min(self.diag_shift_2)}

    @property
    def max(self):
        return {CAP.ROW_SHIFT.value: np.max(self.row_shift), 
                CAP.COLUMN_SHIFT.value: np.max(self.column_shift), 
                CAP.DIAGONAL_SHIFT_1.value: np.max(self.diag_shift_1), 
                CAP.DIAGONAL_SHIFT_2.value: np.max(self.diag_shift_2)}

    @property
    def global_mean(self):
        return np.mean(self.merge_all())

    @property
    def global_std(self):
        return np.std(self.merge_all())

    @property
    def global_min(self):
        return np.min(self.merge_all())

    @property
    def global_max(self):
        return np.max(self.merge_all())
    
