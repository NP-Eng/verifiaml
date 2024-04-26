use hcs_common::{BMMNode, NodeOpsNative, RequantizeBMMFloatNode, Poly, ReLUNode, ReshapeNode};

#[cfg(feature = "test-types")]
use {
    hcs_common::Tensor,
    ark_bn254::Fr,
    hcs_common::Ligero,
    ark_crypto_primitives::sponge::poseidon::PoseidonSponge,
};

use crate::NodeOpsProve;
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

pub fn node_downcast<
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
>(
    node: &dyn NodeOpsNative<i8>,
) -> &dyn NodeOpsProve<F, S, PCS, i8> {
    let node_as_any = node.as_any();

    Option::None
        .or_else(|| {
            node_as_any
                .downcast_ref::<ReLUNode<i8>>()
                .clone()
                .map(|x| x as &dyn NodeOpsProve<F, S, PCS, i8>)
        })
        .or_else(|| {
            node_as_any
                .downcast_ref::<ReshapeNode>()
                .clone()
                .map(|x| x as &dyn NodeOpsProve<F, S, PCS, i8>)
        })
        .or_else(|| {
            node_as_any
                .downcast_ref::<BMMNode<i8>>()
                .clone()
                .map(|x| x as &dyn NodeOpsProve<F, S, PCS, i8>)
        })
        .or_else(|| {
            node_as_any
                .downcast_ref::<RequantizeBMMFloatNode<i8>>()
                .clone()
                .map(|x| x as &dyn NodeOpsProve<F, S, PCS, i8>)
        })
        .expect("No implementor of NodeOpsProve was received")
}

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

    let nodes_proof: Vec<&dyn NodeOpsProve<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8>> = nodes
        .iter()
        .map(|x| node_downcast::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(x.as_ref()))
        .collect();

    assert!(nodes_proof
        .iter()
        .zip(nodes.iter())
        .all(|(x, y)| { x.type_name() == y.type_name() }));
}
