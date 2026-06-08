pub fn parse_config(path: &str) -> Result<Config, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    Ok(Config::from_str(&content))
}

pub struct Config {
    pub name: String,
}

impl Config {
    pub fn from_str(s: &str) -> Self {
        Config { name: s.to_string() }
    }
}
