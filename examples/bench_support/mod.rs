#[derive(Clone, Copy, Debug)]
pub struct MeasurementSummary {
    pub samples: usize,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub spread_percent: f64,
}

impl MeasurementSummary {
    pub fn new(seconds: &[f64]) -> Self {
        assert!(!seconds.is_empty(), "measurement window cannot be empty");
        let mut sorted = seconds.to_vec();
        sorted.sort_by(f64::total_cmp);
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let median = sorted[sorted.len() / 2];
        Self { samples: sorted.len(), median, min, max, spread_percent: (max / min - 1.0) * 100.0 }
    }

    pub fn needs_extended_window(&self) -> bool {
        self.samples >= 3 && self.spread_percent > 3.0
    }
}
