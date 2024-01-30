
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_poly_commit::PolynomialCommitment;

use crate::model::{
    Model,
    Poly,
    nodes::{
        Node,
        reshape::ReshapeNode,
        fc::FCNode,
        relu::ReLUNode,
    },
};

mod parameters;

use parameters::*;

const INPUT_DIMS: &[usize] = &[28, 28];
const OUTPUT_DIMS: &[usize] = &[10];

fn build_simple_perceptron_mnist<F, S, PCS>() -> Model<F, S, PCS> 
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{

    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode<F, S, PCS> = ReshapeNode::new(
        INPUT_DIMS.to_vec(),
        vec![flat_dim],
    );

    let fc: FCNode<F, S, PCS> = FCNode::new(
        WEIGHTS.to_vec(),
        BIAS.to_vec(),
        (flat_dim, OUTPUT_DIMS[0]),
        S_I,
        Z_I,
        S_W,
        Z_W,
        S_O,
        Z_O,
    );

    Model::new(vec![
        Node::Reshape(reshape),
        Node::FC(fc),
    ])
}

#[cfg(test)]
mod tests {
    use ark_ff::PrimeField;
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_poly_commit::PolynomialCommitment;

    use super::*;

    #[test]
    fn run_simple_perceptron_mnist() {
        let model = build_simple_perceptron_mnist::<Fr, DummySponge<Fr>, DummyPC<Fr>>();
        let input = vec![vec![vec![1; 28]; 28]];
        let output = model.evaluate(input);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 1);
        assert_eq!(output[0][0].len(), 10);
        assert_eq!(output[0][0][0], 0);
        assert_eq!(output[0][0][1], 0);
        assert_eq!(output[0][0][2], 0);
        assert_eq!(output[0][0][3], 0);
        assert_eq!(output[0][0][4], 0);
        assert_eq!(output[0][0][5], 0);
        assert_eq!(output[0][0][6], 0);
        assert_eq!(output[0][0][7], 0);
        assert_eq!(output[0][0][8], 0);
        assert_eq!(output[0][0][9], 0);
    }
}