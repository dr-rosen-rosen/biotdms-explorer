# Property Glossary


## Object properties

| Property | Label | Domain | Range | Definition |
|---|---|---|---|---|
| `evid:aggregatesFindingsFrom` | aggregates findings from | `evid:metaAnalysis` | `evid:primaryStudy` | individual studies included in a meta-analysis |
| `evid:hasDependentVariable` | has dependent variable | `evid:EffectSize` | `meas:Measure` | Links an effect size to the measure used as the dependent variable in a given analysis. |
| `evid:hasIndependentVariable` | has independent variable | `evid:EffectSize` | `meas:Measure`, `meas:Manipulation`, `meas:Measure`, `meas:Manipulation` | Links an effect size to the measure or manipulation used as the independent variable in a given analysis. |
| `evid:hasSourceConstruct` | has source construct | `evid:ClassLevelRelationship` | `meas:Construct` | Specifies the source construct for a meta-analytic effect |
| `evid:hasTargetConstruct` | has target construct | `evid:ClassLevelRelationship` | `meas:Construct` | Specifies the target construct for a meta-analytic effect |
| `evid:reportsEffectSize` | reports effect size | `evid:Study` | `evid:EffectSize` | links effect sizes to the studies that report them |
| `evid:reportsStudy` | reports study | `evid:Publication` | `evid:Study` | Publication reporting details of an included study. |
| `evid:summarizesEffectBetween` |  | `evid:metaAnalysis` | `evid:ClassLevelRelationship` |  |
| `meas:hasLevelOfAnalysis` | has level of analysis | `meas:Measure` | `meas:levelOfAnalysis` | Specifies whether the measure is team-level, individual-level, etc. |
| `meas:includesModality` | includes modality | `meas:Measure` | `meas:Modality` | Indicates the source modality or signal the measure relies on. |
| `meas:manipulatesConstruct` | manipulates construct | `meas:Manipulation` | `meas:Construct` | Relates a manipulation to the construct it intends to manipulate. |
| `meas:measuresConstruct` | measures construct | `meas:Measure` | `meas:Construct` | Relates a measure to the construct it captures. |
| `meas:usesAnalyticTechnique` | uses analytic technique | `meas:Measure` | `meas:analyticTechnique` | Specifies the analytic approach used to compute the measure. |
| `meas:usesMethod` | uses method | `meas:Measure` | `meas:Method` | Specifies approach used to compute or derive the measure. |

## Datatype properties

| Property | Label | Domain | Range | Definition |
|---|---|---|---|---|
| `evid:hasDOI` | has DOI | `evid:Publication` | `string` | DOI number or link for publication |
| `evid:hasEffectDomain` | has effect domain | `evid:EffectSize` | `string` | Semantic domain of the effect (e.g., Performance effect, Coordination effect) |
| `evid:hasEffectLevel` | has effect level | `evid:EffectSize` | `string` | level of analysis of the effect (e.g., indiviudal, team, cross-level); pertains to measure level, but also the analysis approach |
| `evid:hasEffectSizeValue` | has effect size value | `evid:EffectSize` | `float` | effect size value |
| `evid:hasFirstAuthor` | has first author | `evid:Publication` | `string` | first author of publication |
| `evid:hasIndividualSampleSize` | individual sample size | `evid:EffectSize` | `float` | number of individuals in the study |
| `evid:hasLowerCI` | has lower CI | `evid:EffectSize` | `float` | lower confidence interval for effect size |
| `evid:hasNotes` | notes | `evid:EffectSize` | `string` | otherwise uncategorized but potentially important information about the effect |
| `evid:hasPubYear` | has publication year | `evid:Publication` | `string` | year of publication |
| `evid:hasPValue` | has p value | `evid:EffectSize` | `float` | p value for effect size |
| `evid:hasSE` | has SE value | `evid:EffectSize` | `float` | standard error value for effect size estimate |
| `evid:hasSignificanceCategory` | has significance category | `evid:EffectSize` | `string` | Categorical significance: NS, Significantly positive, Significantly negative |
| `evid:hasStudyPopulation` | has study population | `evid:Study` | `string` | description of population in the study |
| `evid:hasTeamSampleSize` | team sample size | `evid:EffectSize` | `float` | number of teams in the study |
| `evid:hasUpperCI` | has upper CI | `evid:EffectSize` | `float` | upper confidence interval for effect size |
| `evid:perturbationPhase` | has perturbation phase | `evid:EffectSize` | `string` | whehter the effect is for data captured during perturbation, normal, or the entire performance episode |
| `evid:usesEffectSizeMetric` | type of effect size metric | `evid:EffectSize` | `string` | effect size type used |
| `evid:usesKStudies` | uses k studies | `evid:EffectSize` | `float` | number of studies included in a meta-analytic effect |
| `meas:hasDescription` | has description | `meas:Measure`, `meas:Manipulation`, `meas:Measure`, `meas:Manipulation`, `meas:Measure`, `meas:Manipulation` | `string` | description of what the measure is |
| `meas:hasInterpretation` | has interpretation | `meas:Measure` | `string` | description of how increasing or decreasing values of the scale are interpreted |
| `meas:hasName` | has name | `meas:Measure`, `meas:Manipulation`, `meas:Measure`, `meas:Manipulation`, `meas:Measure`, `meas:Manipulation` | `string` | name of measure |
| `meas:hasScale` | has scale | `meas:Measure` | `string` | describes the scale used including type and range |