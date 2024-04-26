use hcs_common::{
    BMMNode, Model, NodeOpsNative, Poly, ReLUNode, RequantizeBMMFloatNode, ReshapeNode,
};

use crate::{NodeOpsProve, ProvableModel};
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

pub fn node_downcast<
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
>(
    node: &dyn NodeOpsNative<i8>,
) -> Box<dyn NodeOpsProve<F, S, PCS, i8>> {
    let node_as_any = node.as_any();

    Option::None
        .or_else(|| {
            node_as_any
                .downcast_ref::<ReLUNode<i8>>()
                .map(|x| Box::new(x.clone()) as Box<dyn NodeOpsProve<F, S, PCS, i8>>)
        })
        .or_else(|| {
            node_as_any
                .downcast_ref::<ReshapeNode>()
                .map(|x| Box::new(x.clone()) as Box<dyn NodeOpsProve<F, S, PCS, i8>>)
        })
        .or_else(|| {
            node_as_any
                .downcast_ref::<BMMNode<i8>>()
                .map(|x| Box::new(x.clone()) as Box<dyn NodeOpsProve<F, S, PCS, i8>>)
        })
        .or_else(|| {
            node_as_any
                .downcast_ref::<RequantizeBMMFloatNode<i8>>()
                .map(|x| Box::new(x.clone()) as Box<dyn NodeOpsProve<F, S, PCS, i8>>)
        })
        .expect("No implementor of NodeOpsProve was received")
}

#[cfg(all(test, feature = "test-types"))]
mod test {
    use super::*;

    use {
        ark_bn254::Fr, ark_crypto_primitives::sponge::poseidon::PoseidonSponge, hcs_common::Ligero,
        hcs_common::Tensor,
    };

    #[test]
    fn test_downcast() {
        let nodes: Vec<Box<dyn NodeOpsNative<i8>>> = vec![
            Box::new(ReLUNode::<i8>::new(1, 1)),
            Box::new(ReshapeNode::new(vec![1], vec![1])),
            Box::new(BMMNode::<i8>::new(
                Tensor::new(vec![1], vec![1, 1]),
                Tensor::new(vec![1], vec![1]),
                1,
            )),
            Box::new(RequantizeBMMFloatNode::<i8>::new(1, 1.0, 1, 1.0, 1, 1.0, 1)),
        ];

        let nodes_proof: Vec<Box<dyn NodeOpsProve<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8>>> = nodes
            .iter()
            .map(|x| node_downcast::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(x.as_ref()))
            .collect();

        assert!(nodes_proof
            .iter()
            .zip(nodes.iter())
            .all(|(x, y)| { x.type_name() == y.type_name() }));
    }
}
