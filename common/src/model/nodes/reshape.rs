use ark_std::log2;

use crate::{
    model::qarray::{InnerType, QTypeArray},
    QArray,
};

use super::{NodeOpsCommon, NodeOpsNative};

pub struct ReshapeNode {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub padded_input_shape_log: Vec<usize>,
    pub padded_output_shape_log: Vec<usize>,
}

impl<ST> NodeOpsNative<ST, ST> for ReshapeNode
where
    ST: InnerType,
{
    fn shape(&self) -> Vec<usize> {
        self.output_shape.clone()
    }

    fn evaluate(&self, input: &QArray<ST>) -> QArray<ST> {
        // Sanity checks
        // TODO systematise

        assert_eq!(
            *input.shape(),
            self.input_shape,
            "Received input shape does not match node input shape"
        );

        let mut output = input.clone();
        output.reshape(self.output_shape.clone());
        output
    }
}

impl NodeOpsCommon for ReshapeNode {
    fn padded_shape_log(&self) -> Vec<usize> {
        self.padded_output_shape_log.clone()
    }

    fn com_num_vars(&self) -> usize {
        0
    }
}

impl ReshapeNode {
    pub fn new(input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        assert_eq!(
            input_shape.iter().product::<usize>(),
            output_shape.iter().product::<usize>(),
            "Input and output shapes have a different number of entries",
        );

        // TODO does this break the invariant that the product of I and O coincides?
        let padded_input_shape_log = input_shape
            .iter()
            .map(|x| log2(x.next_power_of_two()) as usize)
            .collect();
        let padded_output_shape_log = output_shape
            .iter()
            .map(|x| log2(x.next_power_of_two()) as usize)
            .collect();

        Self {
            input_shape,
            output_shape,
            padded_input_shape_log,
            padded_output_shape_log,
        }
    }
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
