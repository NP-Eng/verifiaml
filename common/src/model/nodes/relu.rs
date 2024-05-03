use ark_std::log2;

use crate::{model::tensor::SmallNIO, Tensor};

use super::{NodeOpsNative, NodeOpsPadded};

// Rectified linear unit node performing x |-> max(0, x).
pub struct ReLUNode<ST> {
    num_units: usize,
    log_num_units: usize,
    pub zero_point: ST,
}

impl<ST> NodeOpsNative<ST, ST> for ReLUNode<ST>
where
    ST: SmallNIO,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.num_units]
    }

    fn evaluate(&self, input: &Tensor<ST>) -> Tensor<ST> {
        // TODO sanity checks (cf. BMM); systematise
        input.maximum(self.zero_point)
    }
}

// impl NodeOpsSnark
impl<ST> NodeOpsPadded<ST, ST> for ReLUNode<ST>
where
    ST: SmallNIO,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.log_num_units]
    }

    fn com_num_vars(&self) -> usize {
        0
    }

    // TODO this is the same as evaluate() for now; the two will likely differ
    // if/when we introduce input size checks
    fn padded_evaluate(&self, input: &Tensor<ST>) -> Tensor<ST> {
        // TODO sanity checks (cf. BMM); systematise
        input.maximum(self.zero_point)
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
