//! Incremental grammar, geometry, and recipe constraints for Factorio output.
//!
//! The schema is emitted beside the corpus tokenizer. It keeps game-specific
//! facts out of the generic model while letting generation mask a token before
//! it creates malformed syntax, an overlap, an out-of-zone footprint, or an
//! illegal machine/recipe pair.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::data::Tokenizer;

#[derive(Debug, Deserialize)]
struct Schema {
    version: String,
    entities: HashMap<String, EntitySchema>,
}

#[derive(Debug, Deserialize)]
struct EntitySchema {
    width: u8,
    height: u8,
    recipes: Vec<String>,
    flow: bool,
}

#[derive(Debug, Clone)]
struct Rule {
    width: u8,
    height: u8,
    recipes: Vec<u16>,
    flow: bool,
}

/// A compiled schema. Token strings are resolved once, before sampling.
#[derive(Debug)]
pub struct Grammar {
    entities: HashMap<u16, Rule>,
    entity_ids: Vec<u16>,
    entity_marker: u16,
    blueprint_end: u16,
    eos: u16,
    xs: Vec<(u16, u8)>,
    ys: Vec<(u16, u8)>,
    directions: Vec<(u16, u8)>,
    flows: Vec<u16>,
}

impl Grammar {
    pub fn load(path: &Path, tokenizer: &Tokenizer) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
        let schema: Schema = serde_json::from_str(&text)
            .with_context(|| format!("cannot parse {}", path.display()))?;
        if schema.version != "factorio-v1" {
            anyhow::bail!("unsupported constraint schema {:?}", schema.version);
        }
        let required = |token: &str| {
            tokenizer
                .token_id(token)
                .with_context(|| format!("constraint token {token:?} is absent from the tokenizer"))
        };
        let mut entities = HashMap::new();
        for (name, body) in schema.entities {
            let id = required(&name)?;
            let recipes =
                body.recipes.iter().map(|recipe| required(recipe)).collect::<Result<_>>()?;
            entities.insert(
                id,
                Rule { width: body.width, height: body.height, recipes, flow: body.flow },
            );
        }
        let mut entity_ids: Vec<u16> = entities.keys().copied().collect();
        entity_ids.sort_unstable();
        Ok(Self {
            entities,
            entity_ids,
            entity_marker: required("<e>")?,
            blueprint_end: required("</bp>")?,
            eos: tokenizer.eos(),
            xs: numbered(tokenizer, "x", 64)?,
            ys: numbered(tokenizer, "y", 64)?,
            directions: numbered(tokenizer, "d", 8)?,
            flows: ["t:input", "t:output"]
                .iter()
                .map(|token| required(token))
                .collect::<Result<_>>()?,
        })
    }

    pub(crate) fn decoder(&self, prompt: &str) -> Decoder<'_> {
        let tokens: Vec<&str> = prompt.split_whitespace().collect();
        let spec = tokens.iter().position(|token| *token == "<spec>");
        let number = |offset: usize| {
            spec.and_then(|at| tokens.get(at + offset))
                .and_then(|token| token.strip_prefix('#'))
                .and_then(|number| number.parse::<u8>().ok())
                .filter(|&number| number > 0)
                .unwrap_or(64)
                .min(64)
        };
        Decoder {
            grammar: self,
            width: number(4),
            height: number(5),
            state: State::Boundary,
            current: None,
            x: None,
            y: None,
            occupied: HashSet::new(),
            placed: 0,
        }
    }
}

fn numbered(tokenizer: &Tokenizer, prefix: &str, count: u8) -> Result<Vec<(u16, u8)>> {
    (0..count)
        .map(|value| {
            let token = match prefix {
                "d" => format!("d{value}"),
                _ => format!("{prefix}{value:02}"),
            };
            tokenizer
                .token_id(&token)
                .with_context(|| format!("constraint token {token:?} is absent from the tokenizer"))
                .map(|id| (id, value))
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Boundary,
    Entity,
    X,
    Y,
    Direction,
    Recipe,
    Flow,
    Done,
}

/// State for one continuation. Occupancy is updated as soon as direction fixes
/// the footprint, before optional recipe and flow fields are sampled.
pub(crate) struct Decoder<'a> {
    grammar: &'a Grammar,
    width: u8,
    height: u8,
    state: State,
    current: Option<u16>,
    x: Option<u8>,
    y: Option<u8>,
    occupied: HashSet<(u8, u8)>,
    placed: usize,
}

impl Decoder<'_> {
    pub(crate) fn allowed(&self) -> Vec<u16> {
        match self.state {
            State::Boundary => {
                let mut out = Vec::with_capacity(2);
                if !self.fitting_entities().is_empty() {
                    out.push(self.grammar.entity_marker);
                }
                if self.placed > 0 {
                    out.push(self.grammar.blueprint_end);
                }
                out
            }
            State::Entity => self.fitting_entities(),
            State::X => self
                .grammar
                .xs
                .iter()
                .filter(|(_, x)| self.position_fits(Some(*x), None))
                .map(|(id, _)| *id)
                .collect(),
            State::Y => self
                .grammar
                .ys
                .iter()
                .filter(|(_, y)| self.position_fits(self.x, Some(*y)))
                .map(|(id, _)| *id)
                .collect(),
            State::Direction => self
                .grammar
                .directions
                .iter()
                .filter(|(_, direction)| self.fits(*direction))
                .map(|(id, _)| *id)
                .collect(),
            State::Recipe => self.rule().recipes.clone(),
            State::Flow => self.grammar.flows.clone(),
            State::Done => vec![self.grammar.eos],
        }
    }

    pub(crate) fn advance(&mut self, token: u16) {
        debug_assert!(self.allowed().contains(&token), "token {token} is not allowed");
        match self.state {
            State::Boundary if token == self.grammar.entity_marker => {
                self.state = State::Entity;
            }
            State::Boundary => self.state = State::Done,
            State::Entity => {
                self.current = Some(token);
                self.state = State::X;
            }
            State::X => {
                self.x = self.value(&self.grammar.xs, token);
                self.state = State::Y;
            }
            State::Y => {
                self.y = self.value(&self.grammar.ys, token);
                self.state = State::Direction;
            }
            State::Direction => {
                let direction = self
                    .value(&self.grammar.directions, token)
                    .expect("allowed direction has a value");
                let recipes = !self.rule().recipes.is_empty();
                let flow = self.rule().flow;
                self.reserve(direction);
                self.placed += 1;
                self.state = if recipes {
                    State::Recipe
                } else if flow {
                    State::Flow
                } else {
                    State::Boundary
                };
            }
            State::Recipe => {
                self.state = if self.rule().flow { State::Flow } else { State::Boundary };
            }
            State::Flow => self.state = State::Boundary,
            State::Done => {}
        }
    }

    fn value(&self, values: &[(u16, u8)], token: u16) -> Option<u8> {
        values.iter().find(|(id, _)| *id == token).map(|(_, value)| *value)
    }

    fn rule(&self) -> &Rule {
        &self.grammar.entities[&self.current.expect("entity was selected")]
    }

    fn fitting_entities(&self) -> Vec<u16> {
        // The prototype table has hundreds of names but only a handful of
        // footprints. Test each rectangular shape once per boundary token.
        let mut shapes = HashMap::new();
        self.grammar
            .entity_ids
            .iter()
            .copied()
            .filter(|id| {
                let rule = &self.grammar.entities[id];
                let shape = (rule.width.min(rule.height), rule.width.max(rule.height));
                *shapes.entry(shape).or_insert_with(|| self.entity_fits(rule))
            })
            .collect()
    }

    fn entity_fits(&self, rule: &Rule) -> bool {
        (0..self.width).any(|x| (0..self.height).any(|y| self.any_direction_fits(rule, x, y)))
    }

    fn position_fits(&self, x: Option<u8>, y: Option<u8>) -> bool {
        let rule = self.rule();
        let xs: Box<dyn Iterator<Item = u8>> = match x {
            Some(value) => Box::new(std::iter::once(value)),
            None => Box::new(0..self.width),
        };
        for x in xs {
            let ys: Box<dyn Iterator<Item = u8>> = match y {
                Some(value) => Box::new(std::iter::once(value)),
                None => Box::new(0..self.height),
            };
            if ys.into_iter().any(|y| self.any_direction_fits(rule, x, y)) {
                return true;
            }
        }
        false
    }

    fn any_direction_fits(&self, rule: &Rule, x: u8, y: u8) -> bool {
        self.rule_fits(rule, x, y, 0)
            || (rule.width != rule.height && self.rule_fits(rule, x, y, 2))
    }

    fn fits(&self, direction: u8) -> bool {
        self.rule_fits(
            self.rule(),
            self.x.expect("x was selected"),
            self.y.expect("y was selected"),
            direction,
        )
    }

    fn rule_fits(&self, rule: &Rule, x: u8, y: u8, direction: u8) -> bool {
        let (width, height) = footprint(rule, direction);
        x.saturating_add(width) <= self.width
            && y.saturating_add(height) <= self.height
            && (x..x + width).all(|xx| (y..y + height).all(|yy| !self.occupied.contains(&(xx, yy))))
    }

    fn reserve(&mut self, direction: u8) {
        let x = self.x.expect("x was selected");
        let y = self.y.expect("y was selected");
        let (width, height) = footprint(self.rule(), direction);
        for xx in x..x + width {
            for yy in y..y + height {
                self.occupied.insert((xx, yy));
            }
        }
        self.x = None;
        self.y = None;
    }
}

fn footprint(rule: &Rule, direction: u8) -> (u8, u8) {
    if matches!(direction, 2 | 6) && rule.width != rule.height {
        (rule.height, rule.width)
    } else {
        (rule.width, rule.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_tokenizer(extra: &[&str]) -> (tempfile::TempDir, Tokenizer) {
        let mut tokens = vec![
            "<unk>".to_owned(),
            crate::data::tokenizer::EOS.to_owned(),
            "<e>".to_owned(),
            "</bp>".to_owned(),
            "t:input".to_owned(),
            "t:output".to_owned(),
        ];
        tokens.extend((0..64).map(|value| format!("x{value:02}")));
        tokens.extend((0..64).map(|value| format!("y{value:02}")));
        tokens.extend((0..8).map(|value| format!("d{value}")));
        tokens.extend(extra.iter().map(|token| (*token).to_owned()));
        let vocab: serde_json::Map<String, serde_json::Value> =
            tokens.iter().enumerate().map(|(id, token)| (token.clone(), id.into())).collect();
        let document = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{
                "id": 1,
                "content": crate::data::tokenizer::EOS,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }],
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        let tokenizer = Tokenizer::load(&path).unwrap();
        (dir, tokenizer)
    }

    fn grammar() -> Grammar {
        let rule = Rule { width: 2, height: 2, recipes: vec![50, 51], flow: false };
        Grammar {
            entities: HashMap::from([(10, rule)]),
            entity_ids: vec![10],
            entity_marker: 1,
            blueprint_end: 2,
            eos: 3,
            xs: (0..4).map(|value| (20 + value as u16, value)).collect(),
            ys: (0..4).map(|value| (30 + value as u16, value)).collect(),
            directions: (0..8).map(|value| (40 + value as u16, value)).collect(),
            flows: vec![60, 61],
        }
    }

    #[test]
    fn schema_and_tokenizer_must_match_before_sampling_starts() {
        let (dir, tokenizer) = word_tokenizer(&["assembling-machine-1", "r:none"]);
        let path = dir.path().join("constraints.json");
        fs::write(
            &path,
            r#"{
                "version": "factorio-v1",
                "entities": {
                    "assembling-machine-1": {
                        "width": 3,
                        "height": 3,
                        "recipes": ["r:none", "r:missing"],
                        "flow": false
                    }
                }
            }"#,
        )
        .unwrap();

        let error = Grammar::load(&path, &tokenizer).unwrap_err();

        assert!(error.to_string().contains(r#""r:missing" is absent"#));
    }

    #[test]
    fn syntax_recipes_zone_and_occupancy_are_masked_incrementally() {
        let grammar = grammar();
        let mut decoder = grammar.decoder("<bp> <spec> k:x r:none #2 #4 #4 </spec>");
        assert_eq!(decoder.allowed(), vec![1]);
        decoder.advance(1);
        decoder.advance(10);
        assert!(!decoder.allowed().contains(&23)); // 2x2 cannot start at x03 in a 4-wide zone.
        for token in [20, 30, 40] {
            decoder.advance(token);
        }
        assert_eq!(decoder.allowed(), vec![50, 51]);
        decoder.advance(50);
        assert_eq!(decoder.allowed(), vec![1, 2]);

        // A second 2x2 entity cannot begin at the occupied top-left tile.
        decoder.advance(1);
        decoder.advance(10);
        decoder.advance(20);
        assert!(!decoder.allowed().contains(&30));
    }
}
