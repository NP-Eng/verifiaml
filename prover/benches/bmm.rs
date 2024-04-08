use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hcs_common::{get_model, get_model_output, run_python};

fn tf_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("tf_inference");
    run_python(|py| {
        let model = get_model(py, "QTwoLayerPerceptron");
        group.bench_with_input(
            BenchmarkId::new("fully connected layer", "mnist"),
            &(),
            |b, _| {
                b.iter(|| get_model_output(py, &model, None));
            },
        );
    });
}

criterion_group!(benches, tf_inference);
criterion_main!(benches);
