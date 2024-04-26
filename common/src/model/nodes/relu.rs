use ark_std::log2;

use crate::{model::tensor::SmallNIO, NIOTensor};

use super::{NodeOpsNative, NodeOpsPadded};

// Rectified linear unit node performing x |-> max(0, x).
#[derive(Clone)]
pub struct ReLUNode<ST> {
    num_units: usize,
    log_num_units: usize,
    pub zero_point: ST,
}

impl<ST> NodeOpsNative<ST> for ReLUNode<ST>
where
    ST: SmallNIO,
{
    fn shape(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![self.num_units], vec![self.num_units])
    }

    fn evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST> {
        let input = input.ref_small();

        // Sanity checks
        self.assert_valid_input(input.shape());

        NIOTensor::S(input.maximum(self.zero_point))
    }

    fn type_name(&self) -> &'static str {
        "ReLU"
    }
}

// impl NodeOpsSnark
impl<ST> NodeOpsPadded<ST> for ReLUNode<ST>
where
    ST: SmallNIO,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.log_num_units]
    }

    fn padded_shape(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![1 << self.log_num_units], vec![1 << self.log_num_units])
    }

    // TODO this is the same as evaluate() for now; the two will likely differ
    // if/when we introduce input size checks
    fn padded_evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST> {
        // TODO sanity checks (cf. BMM); systematise
        NIOTensor::S(input.ref_small().maximum(self.zero_point))
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
