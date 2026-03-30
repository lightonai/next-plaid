use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fastkmeans_rs::{FastKMeans, KMeansConfig};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Duration;

fn benchmark_kmeans_varying_samples(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_samples");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let n_features = 128;
    let k = 50;
    let sample_sizes = [1_000, 5_000, 10_000];

    for n_samples in sample_sizes.iter() {
        group.throughput(Throughput::Elements(*n_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            n_samples,
            |b, &n_samples| {
                let data = Array2::random((n_samples, n_features), Uniform::new(-1.0f32, 1.0));
                let config = KMeansConfig {
                    k,
                    max_iters: 5,
                    tol: 1e-8,
                    seed: 42,
                    max_points_per_centroid: None,
                    chunk_size_data: 51_200,
                    chunk_size_centroids: 10_240,
                    verbose: false,
                };

                b.iter(|| {
                    let mut kmeans = FastKMeans::with_config(config.clone());
                    kmeans.train(black_box(&data.view())).unwrap();
                    kmeans
                });
            },
        );
    }
    group.finish();
}

fn benchmark_kmeans_varying_clusters(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_clusters");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let n_samples = 5_000;
    let n_features = 128;
    let cluster_counts = [10, 50, 100];

    for k in cluster_counts.iter() {
        group.throughput(Throughput::Elements(*k as u64));
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k| {
            let data = Array2::random((n_samples, n_features), Uniform::new(-1.0f32, 1.0));
            let config = KMeansConfig {
                k,
                max_iters: 5,
                tol: 1e-8,
                seed: 42,
                max_points_per_centroid: None,
                chunk_size_data: 51_200,
                chunk_size_centroids: 10_240,
                verbose: false,
            };

            b.iter(|| {
                let mut kmeans = FastKMeans::with_config(config.clone());
                kmeans.train(black_box(&data.view())).unwrap();
                kmeans
            });
        });
    }
    group.finish();
}

fn benchmark_kmeans_varying_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_dimensions");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let n_samples = 2_000;
    let k = 20;
    let dimensions = [32, 128, 256];

    for n_features in dimensions.iter() {
        group.throughput(Throughput::Elements(*n_features as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_features),
            n_features,
            |b, &n_features| {
                let data = Array2::random((n_samples, n_features), Uniform::new(-1.0f32, 1.0));
                let config = KMeansConfig {
                    k,
                    max_iters: 5,
                    tol: 1e-8,
                    seed: 42,
                    max_points_per_centroid: None,
                    chunk_size_data: 51_200,
                    chunk_size_centroids: 10_240,
                    verbose: false,
                };

                b.iter(|| {
                    let mut kmeans = FastKMeans::with_config(config.clone());
                    kmeans.train(black_box(&data.view())).unwrap();
                    kmeans
                });
            },
        );
    }
    group.finish();
}

fn benchmark_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_predict");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let n_train = 5_000;
    let n_features = 128;
    let k = 50;
    let predict_sizes = [1_000, 5_000];

    // Pre-train the model
    let train_data = Array2::random((n_train, n_features), Uniform::new(-1.0f32, 1.0));
    let config = KMeansConfig {
        k,
        max_iters: 10,
        tol: 1e-8,
        seed: 42,
        max_points_per_centroid: None,
        chunk_size_data: 51_200,
        chunk_size_centroids: 10_240,
        verbose: false,
    };
    let mut kmeans = FastKMeans::with_config(config);
    kmeans.train(&train_data.view()).unwrap();

    for n_predict in predict_sizes.iter() {
        group.throughput(Throughput::Elements(*n_predict as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_predict),
            n_predict,
            |b, &n_predict| {
                let test_data = Array2::random((n_predict, n_features), Uniform::new(-1.0f32, 1.0));

                b.iter(|| kmeans.predict(black_box(&test_data.view())).unwrap());
            },
        );
    }
    group.finish();
}

fn benchmark_colbert_like(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_colbert");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    // ColBERT-like configuration (smaller for faster benchmarks)
    let n_samples = 100_000;
    let k = 1024;

    group.bench_function("100k_samples_1k_clusters", |b| {
        let data = Array2::random((n_samples, 128), Uniform::new(-1.0f32, 1.0));
        let config = KMeansConfig {
            k,
            max_iters: 5,
            tol: 1e-8,
            seed: 42,
            max_points_per_centroid: Some(256),
            chunk_size_data: 51_200,
            chunk_size_centroids: 10_240,
            verbose: false,
        };

        b.iter(|| {
            let mut kmeans = FastKMeans::with_config(config.clone());
            kmeans.train(black_box(&data.view())).unwrap();
            kmeans
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_kmeans_varying_samples,
    benchmark_kmeans_varying_clusters,
    benchmark_kmeans_varying_dimensions,
    benchmark_predict,
    benchmark_colbert_like,
);

criterion_main!(benches);
