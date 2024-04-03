use crate::{Node, NodeCommitment};

// In this trait, we closely follow the notation of the LegoSNARK paper
trait CPSNARK {
    const ARITY: usize;

    type CommitmentKey; // ck used to commit and open commitments
    type EvaluationKey; // ek used to produce proofs
    type VerificationKey; // vk used to verify proofs

    type Commitment; // C
    type Opening; // O (a.k.a. opening hint)

    type Instance; // D_x
    type Value; // D containing D_i for 1 <= i <= arity
    type Witness; // D_omega

    type Proof; // pi

    // A description of a relation on Instance x (Value x Witness)
    // which is used to generate the evaluation and verification keys
    type Relation;

    fn key_gen(
        ck: &Self::CommitmentKey,
        r: &Self::Relation,
    ) -> (Self::EvaluationKey, Self::VerificationKey);

    fn prove(
        ek: &Self::EvaluationKey,
        x: &Self::Instance,
        commitments: [&Self::Commitment; Self::ARITY],
        values: [&Self::Value; Self::ARITY],
        openings: [&Self::Opening; Self::ARITY],
        omega: &Self::Witness,
    ) -> Self::Proof;

    fn verify_proof(
        vk: &Self::VerificationKey,
        x: &Self::Instance,
        commitments: [&Self::Commitment; Self::ARITY],
        pi: &Self::Proof,
    ) -> bool;
}

// Taylored version of CPSNARK adapted to node proofs
trait NodeCPSNARK {
    type CommitmentKey; // ck used to commit and open commitments to parameters, inputs and outputs
    type EvaluationKey; // ek used to produce proofs
    type VerificationKey; // vk used to verify proofs

    type ParamCommitment; // C_1
    type ParamHint; // O_1 (a.k.a. opening hint)
    type ParamValue; // D_1

    type IOCommitment; // C_2, space for the input as well as output
    type IOHint; // O_2 (a.k.a. opening hint)
    type IOValue; // D_2

    type Instance; // D_x

    type Proof; // pi

    fn key_gen(
        ck: &Self::CommitmentKey,
        node: &Self,
    ) -> (Self::EvaluationKey, Self::VerificationKey);

    fn prove(
        ek: &Self::EvaluationKey,
        instance: &Self::Instance,
        param_commitment: &Self::ParamCommitment,
        input_commitment: &Self::IOCommitment,
        ouput_commitment: &Self::IOCommitment,
        param_value: &Self::ParamValue,
        input_value: &Self::IOValue,
        output_value: &Self::IOValue,
        param_hint: &Self::ParamHint,
        input_hint: &Self::IOHint,
        output_hint: &Self::IOHint,
        // no non-committed witness omega
    ) -> Self::Proof;

    fn verify_proof(
        vk: &Self::VerificationKey,
        instance: &Self::Instance,
        param_commitment: &Self::ParamCommitment,
        input_commitment: &Self::IOCommitment,
        ouput_commitment: &Self::IOCommitment,
        pi: &Self::Proof,
    ) -> bool;
}

enum ParamOrIO<ParamType, IOType> {
    Param(ParamType),
    IO(IOType),
}

fn get_param_input_output<ParamType, IOType>(
    array: [&ParamOrIO<ParamType, IOType>; 3],
) -> (&ParamType, &IOType, &IOType) {
    match array {
        [ParamOrIO::Param(p), ParamOrIO::IO(i), ParamOrIO::IO(o)] => (p, i, o),
        _ => panic!("Invalid structure"),
    }
}

impl<NCPS: NodeCPSNARK> CPSNARK for NCPS {
    const ARITY: usize = 3;

    type CommitmentKey = NCPS::CommitmentKey;
    type EvaluationKey = NCPS::EvaluationKey;
    type VerificationKey = NCPS::VerificationKey;

    type Commitment = ParamOrIO<NCPS::ParamCommitment, NCPS::IOCommitment>;

    type Opening = ParamOrIO<NCPS::ParamHint, NCPS::IOHint>;

    type Value = ParamOrIO<NCPS::ParamValue, NCPS::IOValue>;

    type Instance = NCPS::Instance;

    type Witness = ();

    type Proof = NCPS::Proof;

    type Relation = Self;

    fn key_gen(
        ck: &Self::CommitmentKey,
        r: &Self::Relation,
    ) -> (Self::EvaluationKey, Self::VerificationKey) {
        NCPS::key_gen(ck, r)
    }

    fn prove(
        ek: &Self::EvaluationKey,
        x: &Self::Instance,
        commitments: [&Self::Commitment; Self::ARITY],
        values: [&Self::Value; Self::ARITY],
        openings: [&Self::Opening; Self::ARITY],
        omega: &Self::Witness,
    ) -> Self::Proof {
        let (param_commitment, input_commitment, output_commitment) =
            get_param_input_output(commitments);
        let (param_value, input_value, output_value) = get_param_input_output(values);
        let (param_hint, input_hint, output_hint) = get_param_input_output(openings);

        NCPS::prove(
            ek,
            x,
            &param_commitment,
            &input_commitment,
            &output_commitment,
            &param_value,
            &input_value,
            &output_value,
            &param_hint,
            &input_hint,
            &output_hint,
        )
    }

    fn verify_proof(
        vk: &Self::VerificationKey,
        x: &Self::Instance,
        commitments: [&Self::Commitment; Self::ARITY],
        pi: &Self::Proof,
    ) -> bool {
        let (param_commitment, input_commitment, output_commitment) =
            get_param_input_output(commitments);

        NCPS::verify_proof(
            vk,
            x,
            &param_commitment,
            &input_commitment,
            &output_commitment,
            pi,
        )
    }
}
