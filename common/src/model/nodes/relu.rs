use ark_std::log2;

use crate::model::qarray::{InnerType, QTypeArray};

use super::{NodeOpsCommon, NodeOpsNative};

// Rectified linear unit node performing x |-> max(0, x).
pub struct ReLUNode<ST> {
    num_units: usize,
    log_num_units: usize,
    pub zero_point: ST,
}

impl<ST, LT> NodeOpsNative<ST, LT> for ReLUNode<ST>
where
    ST: InnerType,
    LT: InnerType + From<ST>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.num_units]
    }

    fn evaluate(&self, input: &QTypeArray<ST, LT>) -> QTypeArray<ST, LT> {
        // TODO sanity checks (cf. BMM); systematise
        let input = input.ref_small();

        QTypeArray::S(input.maximum(self.zero_point))
    }
}

// impl NodeOpsSnark
impl<ST> NodeOpsCommon for ReLUNode<ST> {
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.log_num_units]
    }

    fn com_num_vars(&self) -> usize {
        0
    }
}

impl<ST> ReLUNode<ST> {
    pub fn new(num_units: usize, zero_point: ST) -> Self {
        let log_num_units = log2(num_units.next_power_of_two()) as usize;

        Self {
            num_units,
            log_num_units,
            zero_point,
        }
    }
}
