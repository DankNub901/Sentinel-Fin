use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3::prelude::*;
use std::collections::HashSet;

// #14: Activity Density
#[polars_expr(output_type=Float64)]
fn calc_activity_density(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.u32()?; 
    if ca.is_empty() { return Ok(Series::full_null(s.name(), 1, &DataType::Float64)); }
    let count = ca.len() as f64;
    let min = ca.min().unwrap_or(0) as f64;
    let max = ca.max().unwrap_or(0) as f64;
    let density = count / (max - min + 1e-9);
    Ok(Float64Chunked::full(s.name(), density, ca.len()).into_series())
}

// #16: Structuring Flag (New Pair)
#[polars_expr(output_type=Boolean)]
fn is_new_pair(inputs: &[Series]) -> PolarsResult<Series> {
    let names_orig = inputs[0].str()?;
    let names_dest = inputs[1].str()?;
    let mut seen = HashSet::new();
    let mut out = BooleanChunkedBuilder::new("is_new_pair", names_orig.len());
    for (o, d) in names_orig.into_iter().zip(names_dest.into_iter()) {
        if let (Some(orig), Some(dest)) = (o, d) {
            let pair = format!("{}_{}", orig, dest);
            out.append_value(!seen.contains(&pair));
            seen.insert(pair);
        } else { out.append_null(); }
    }
    Ok(out.finish().into_series())
}

// #13: Decayed Velocity (Stateful)
#[polars_expr(output_type=Float64)]
fn decayed_velocity(inputs: &[Series]) -> PolarsResult<Series> {
    let steps = inputs[0].u32()?;
    let mut velocity: f64 = 0.0;
    let mut last_step: u32 = 0;
    
    // Standard Rust Vec: Very stable, avoids Builder-related compilation errors
    let mut values = Vec::with_capacity(steps.len());

    for step_opt in steps.into_iter() {
        if let Some(step) = step_opt {
            let delta_t = (step - last_step) as f64;
            velocity = (velocity * (-0.1 * delta_t).exp()) + 1.0;
            values.push(Some(velocity));
            last_step = step;
        } else {
            values.push(None);
        }
    }
    
    let out = Float64Chunked::from_iter_options("decayed_vel", values.into_iter());
    Ok(out.into_series())
}

// #15: Similarity Score
#[polars_expr(output_type=Float64)]
fn get_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let amt = inputs[0].f64()?;
    let fraud_centroids = [10000.0, 50000.0, 100000.0];
    
    let out: Float64Chunked = amt.apply(|val| {
        val.map(|v| {
            fraud_centroids.iter()
                .map(|c| (v - c).abs())
                .fold(f64::INFINITY, f64::min)
        })
    });
    Ok(out.into_series())
}

#[pymodule]
fn rust_engine(_py: Python, _m: &PyModule) -> PyResult<()> {
    Ok(())
}