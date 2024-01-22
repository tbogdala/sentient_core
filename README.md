# Sentient Core Development Preview v0.3.0

A terminal style user interface to chat with AI characters.

![Demo of sentient_core in action showing interaction with the default chatbot, Vox](https://github.com/tbogdala/sentient_core/blob/main/screenshots/DemoScreencast-01.gif)

## Features

The project is in its early stages, so some basic features may still be missing.


### Overall
- [x] GGUF Llama models or any model that [Llama.cpp](https://github.com/ggerganov/llama.cpp) supports.
- [x] GPU layer offloading for accelleration of local text generation
- [x] Optionally use koboldcpp as a backend for text generation
- [ ] Streaming-mode for text inference


### Chatting

- [x] loading text-generation-webui yaml files for characters
- [x] log scrolling ('j'/'k' key commands or up/down arrows)
- [x] create replies to the bot ('r' key command)
- [x] switch between parameter configuration sets ('p' key command, then 'h'/'l' (or left/right arrows) to swtich between)
- [x] saving and switching between multiple chat logs for a character ('ctrl-n' in the log selector menu creates a new log)
- [x] duplicating chat logs ('ctrl-d' in the log selector menu creates a duplcicate copy of the log)
- [x] regenerate ('ctrl+r' key command)
- [x] continue ('crtl-t' key command)
- [x] additional generation ('ctrl-y' key command)
- [ ] regenerate? (attempt a new text generation in a popup to be accepted ot rejected)
- [x] edit the 'current context' for the chatlog ('o' key command)
- [x] edit the 'user description' for the chatlog ('ctrl-o' key command)
- [x] edit ('e' key command) [Note: basic support]
- [x] remove currently selected chatlog entry ('ctrl-x' key command)
- [x] colorized log output
- [ ] resizable width of text display
- [x] stops the AI reponses at your display name's tag.
- [x] multiline input is supported by ending a line with "\n" and hitting enter.
- [x] 'multi-chat' mode ('m' key) allowing the user to ('r') reply as themselves or click a number 1-9 to reply
      as another participant. The '1' key is bound to the character owning the chatlog file.
- [ ] command interface to input special instructions such as getting or setting variables used in text generation (in active development)
- [x] emotional boosting characters to be biased towards specified emotions (by setting the variable 'emotional_boosts' or shorthand, 'eb')


### Configuration

- [x] Configurable sets of models and prompt templates
- [x] Sets of hyperparameters for text inference
- [x] Set the context length for the models
- [x] Configurable settings for both CPU and GPU inference
- [x] Configurable number of layers to offload to gpu
- [ ] Use characters and logs in a standards compliant location (XDG or equivalent)
- [x] configurable justifaction: left, right center
- [x] configurable display name
- [x] configurable colors in the chatlog for user, bot, normal text and quoted text
- [x] user descriptions stored in the chatlog so they can change for the context of the log
- [x] memory file support (AKA 'lorebooks' or 'world info')


### General Features

- [x] export chatlog as a dataset for finetuning. currently only exports input-output format JSONL (ctrl-o in character log select)
- [x] vector embedding support for sentence similarity testing against the chatlog (only cuda accelleration for now)
- [ ] spellchecker integration
- [ ] import/export plaintext logs
- [x] export datasets from chatlogs


## Requirements

* Windows users must install [CUDA](https://developer.nvidia.com/cuda-downloads) from Nvidia. Version 11.8 has been tested to work.
* Linux binaries were built with CUDA 11.
* Mac uses metal so no further libraries are needed.

It's suggested that you have at least 8GB of VRAM in your graphics cards so that you can run a 7B parameter
model completely GPU accellerated. While it's possible to offload only some layers, it will be far slower.


## Installation

1) Unzip the binaries from the release archives published here or build from source (see below).
2) Edit the supplied `config.yaml` file to change `display_name` and `use_gpu`, `batch_size` and `thread_count` appropriately.
3) Make sure to configure a model in `config.yaml` and name it.
4) Run the app from the command line: `./sentient_core -m <model_name>`

Quick tip: all screens besides the main menu have built in help with the `?` key. Escape will back out
of any message box or view.

The sample `config.yaml` file uses a quantized version of Nous-Hermes-13B by TheBloke and can 
be [downloaded from this page](https://huggingface.co/TheBloke/Nous-Hermes-13B-GGUF). Specifically, the configuration file
mentions this [file](https://huggingface.co/TheBloke/Nous-Hermes-13B-GGUF/resolve/main/Nous-Hermes-13B.Q4_K_M.gguf). 
You can create the `models/TheBloke_/_Nous-Hermes-13B-GGUF` folders next to the executable and download
the file to that new folder. After that running `./sentient_core -m nous-hermes-13b` should work.


## Configuration

The `config.yaml` shipped with the application is an example starting point, and
contains sampler presets that are defaults in other apps. The comments in the file
should help show what options are available an what they do.

Every user should change the `display_name` field to whatever they want to be 
called in the chatlog. Also, make sure to uncomment `use_gpu: true` and `batch_size: 512` if you 
want GPU accelleration with cuda or metal. Also make sure to set `thread_count` to the number of 
cores your CPU has if you're not offloading all layers.

After that, make sure to configure the models to use. below you can see an example configuration block
for a model we're giving the name `nous-hermes-13b` (defined in the `name` field). This is
what you'd pass with the `-m` command line argument to load the model at startup.

The filepath to the model is relative and defined in the `path` field. Alternatively, a
`remote_path` field could be set to say `http://localhost:5001` to use KoboldCpp as a backend.

Other miscelaneous settings are the `context_size` which you can use to control how many
tokens to try and send to the LLM. The `text_to_token_ratio_prediction` parameter in the configuration
file is related to this. When multiplied together, that's how many characters the software budgets.

If `use_gpu` is set to `true`, you can control the number of layers to offload with the 
`gpu_layer_count` field. Overshooting the number of layers is fine to force the offloading of all layers.

Lastly, make sure to define a `prompt_instruct_template`. You can see an example below that includes
Alpaca-style instruct text such as `### Instruction` and `### Response`, but most models respond
without the instruct tokens too. Experiment to find out what layout works the best for you.

Multiple models can be defined, and in multi-chat mode, other participants can even use different
models than the main character!

```yaml
# ...<snip>...
models:
  - name: "nous-hermes-13b"
    path: "models/TheBloke_Nous-Hermes-Llama2-GGUF/nous-hermes-llama2-13b.Q4_K_M.gguf"
    context_size: 4096
    gpu_layer_count: 100
    prompt_instruct_template: |- 
      ### Instruction:
      Continue the chat dialogue below. Write a single reply for the character named "<|character_name|>".
      <|character_description|>
      <|user_description|>
      <|character_context|>
      <|chat_history|>
      
      ### Response: 
      <|character_name|>: 
# ...<snip>...
```

### Advanced Configuration

Models can be configured to use an extended context size by adjusting their rope scale using the model's
`rope_scale` parameter. This mirrors's [the llama.cpp documentation](https://github.com/ggerganov/llama.cpp/tree/master/examples/main#extended-context-size)
for the similarly named parameter. If you wish to adjust the frequency for whatever reason,
that can be set with the `rope_freq` parameter.


## Interface Reference

Typing the `?` character while in any other view than the main menu should bring up
a command reference window. In general the `enter` key will confirm input or message box,
and the `esc` key will back out of any message boxes or views.


## Multi-Chat Mode

Pressing 'm' in the chat view will toggle Multi-Chat Mode where AI responses are not automatically generated
to your own replies. This allows for total control of the flow of conversation between multiple AI characters.
You still write your own replies by hitting 'r' to respond, and can trigger the primary character to respond
with 'ctrl-y', but you can also use the number keys 1-0 which will generate a response for the matching
character. Key '1' is bound to to the primary character, the one 'owning' the chatlog, and starting with 
the '2' key, they go in order listed in the chatlog.

To configure the log file for multi-chat mode, add an `other_participants` field and assign it an array of
objects, with each object having a `character_filepath` field that is assigned a relative filepath string that
should point to another character yaml file. 

In this example below, `char1.yaml` would be two directories up from this log file, which typically is where
all of your character yaml files are stored. `char2.yaml` shows that you can place the character files
right in the log directory to keep it organized that way if desired.

```json
{
  "version": 1,
  "other_participants": [
    {
      "character_filepath": "../../char1.yaml"
    },
    {
      "character_filepath": "char2.yaml"
    }
  ],
  # ...<snip>...
```

Additionally, you can specify a model configuration to use to generate the response for the other
participants, allowing for custom finetunes to be used for each character. 

```json
  # ...<snip>...
  "other_participants": [
    {
      "model_config_name": "llama2-13b",
      "character_filepath": "../../char1.yaml"
    }
  ],
  # ...<snip>...
```


## Advanced Template Options

### Character Memory Files

A very handy feature to have is the Memory File support for characters. This is a separate JSON file that consists
simply of a `memories` field that is an array of objects. Each of these objects has a `key` and `value` field. The
`key` fields are search for in the relevant chatlog entries (Note: Currently only the last chatlog entry), and if
it occurs, then the `value` field's text will get included in the `<|memory_matches|>` field in the prompt for the model.

An example of a memory file would look like this:

```json
{
    "memories": [
      { 
            "key": "SentientCore",
            "value": "SentientCore is the AI software that is the software currently running this holodeck and allows for all of the human and AI interaction."
      }
    ]
}
```

#### Usage

Using this example Memory File, if the user replies to the character and mentions 'SentientCore' (case-sensitive) anywhere 
in the reply, the value text will get added to the `<|memory_matches|>` template parameter.

In order to have the Memory File actually loaded and searched, **it needs to be manually included in the chat log**.
In the `log.json` file for the chatlog, add the following field to the root object of the chatlog (parallel to other 
fields like `current_context` and `items`):

```json
  # ...<snip>...
  "memory_files": [
    "memories.json"
  ],
  # ...<snip>...
```

Multiple memory files are supported. Multiple entries for the same `key` are supported and all will get potentially included.

In order to prevent the memories from taking over all the space available in the context, there's a built in limit of 10% of
the context. If a memory isn't estimated to fit in that amount of tokens, it won't be included and further memory processing 
is halted. The size of this limit can be configured in the main `config.yaml` file using the `memory_max_context_percentage`
field, which defaults to `0.1` for 10%. Setting it to `0.05` would limit it to 5% instead, for example.

### Emotional Boosts

Another prompt template idea is the notion of specifying 'emotional boosts' to help flavor the generated text. This
way the character file doesn't need to change, only the desired changes to emotions to help direct the LLM
neural net in the direction of your choosing. A default set of emotional boosts can be defined in the character YAML files
by adding the `emotional_boosts` field and setting it to a string. An example of this has been added to the default
character of SentientCore: Vox!

```yaml
emotional_boosts: "Quite Empathetic, Quite Relaxed, Slightly Confident"
```

When chatting with a character, the emotional boosts can be changed using the Editor Command Interface by setting
the `emotional_boosts` variable, or using it's shortcut: `eb`.



## Editor Command Interface

You can enter commands to control the chat interface by bringing up the command interface box with the '/' key.
which will bring up a window in which to type the command. This feature is in active development and will be 
expanded on further. 

In general, after specifying a command, an optional character number to use as the target can be supplied. If
this is omitted, the main owning character of the chatlog is used. Otherwise, in multi-chat mode, you can specify
the main character as `1` and other participating characters in order of their definition in the chatlog file, starting
with number `2`.


### Command Reference

Below are the commands the command interface window will accept:

#### get

syntax: `get <character_num>? <variable_name>`

This command returns the value of the variable in a message box. If the variable is something associated 
with a character, it will default to `character_num` 1 unless specified otherwise. 

#### set

syntax: `set <character_num>? <variable_name> ...`

This command set the value of the variable in a message box. If the variable is something associated 
with a character, it will default to `character_num` 1 unless specified otherwise. 

### Variable Reference

#### emotional_boosts ("eb")

This variable is accessed from the specified character. For example `set 1 emotional_boosts Very Calm` 
will set the `emotional_boosts` field of that character to `"Very Calm"`.



## Building from source

The way the project is currently checked into github, `cuda` is enabled as a feature
for the `rust-llama.cpp` dependency. If that is not compatible with your needs, you will need
to change the `Cargo.toml` file before building, or pass `--no-default-features` on the command line. 
For example, you'll need to enable the `metal` feature instead for hardware accelleration on macos. 

The feature `sentence_similarity` uses the [Candle library](https://github.com/huggingface/candle)
to load the BERT models for generating vector embeddings. As is, the project should build 
including `sentence_similarity_cuda` as a default feature, allowing accelleration for cuda compatible devices.
When other backends become available, different `sentence_similarity` feature groups will be added, but
it is recommended to just disable the feature if it cannot be hardware accellerated. Currently it
is known not to build on Windows with Cuda accelleration.


```bash
git clone https://github.com/tbogdala/sentient_core.git
cd sentient_core
cargo build --release
```

Then the project can be run through cargo by specifying a command like:

```bash
# run using a model file without configuration
cargo run --release -- -m ~/llm-models/TheBloke_Nous-Hermes-13B-GGML/nous-hermes-13b.ggmlv3.q4_K_M.bin

# or run a configured model by name
cargo run --release -- -m nous-hermes-13b 

# or you can send stderr to a different terminal so you can get debug output.
# you can get the device path for this by running `tty` in the target terminal 
# (also illustrating custom configuration file specification):
RUST_LOG=debug cargo run --release -- -m nous-hermes-13b -c config.yaml 2> /dev/pts/1
```

## Compatible Models

The current backend uses my fork of [rust-llama.cpp](https://github.com/tbogdala/rust-llama.cpp) 
which wraps and embeds [llama.cpp](https://github.com/ggerganov/llama.cpp) into the binary. Currently,
this means that sentient_core supports all of the models that llama.cpp does. At present, that means
that for Llama models, they must be in the GGUF model format as the library doesn't support GGML and previous
versions.


## Using KoboldCpp Backend

[KoboldCpp](https://github.com/LostRuins/koboldcpp) can be used as a backend for text inference instead
of using the built-in [llama.cpp](https://github.com/ggerganov/llama.cpp) support.

To accomplish this, instead of specifying `path` in a named model configuration, specify 
`remote_path` instead, which should look something like `http://localhost:5001` 
(note: no trailing slash, but port number is included).


## Creating New Characters

The method to add a new character is simple: In `characters`, copy `Vox.yaml` and rename it to match the name of your character.
From there, the software should create a default chatlog for that character.

The following templates are supported in prompt templates on the models in the `config.yaml`:

* `<|character_description|>`: The character description from the character's yaml file.
* `<|user_description|>`: If the `user_description` field from the chatlog is set, that value be used.
* `<|current_context|>`: The `current_context` field from the chatlog, which is populated initially with the `context` from the character file.
* `<|similar_sentences|>`: The sentence similary results from running vector embedding searches through the log. Only include this if the `sentence_similarity` feature is enabled or else no substitution will happen.
* `<|memory_matches|>`: Replaced with any 'values' from memories where the 'key' is found in the last message. Multiple memories can stack even if they have the same 'key'.
* `<|emotional_boosts|>`: Replaced with the `emotional_boosts` value from the character configuration file if present; can be updated with the Command Interface.
* `<|character_name|>`: The name of the current character to generate a response for.
* `<|user_name|>`: The name of the user, pulled from the `display_name` field in the `config.yaml` file.


## Sentence Simlarity with Vector Embeddings

In order to use vector embeddings, you will need to add an `embedding_model` section to your `config.yaml` file.
It can look something like this:

```yaml
embedding_model:
  dir_path: "models/bge-large-en-v1.5"
  token_cutoff_limit: 512
  use_cpu: false
  query_pretext: "Represent this sentence for searching relevant passages: "
  encode_pretext: "Represent this sentence for searching relevant passages: "
```

The `dir_path` should point to a compatible BERT embedding model. The [bge line of models from BAII](https://huggingface.co/BAAI/bge-large-en-v1.5)
have worked well for me, as has [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Note that the query and encode pretext *do not* need to be defined, but are there as options to prepend text 
to aid the embedding model genenerate better data if the model needs it.

With the `embedding_model` section of `config.yaml` defined, you can now include `<|similar_sentences|>` into
your prompt template to have them replaced with past chatlog items that are detected to be similar to the
last response being presented to the LLM. The number of responses can be configured via the `similar_sentence_count`
parameter in the model configuration.

Currently, this is implemented with [Candle](https://github.com/huggingface/candle), but that might change
if these models get support in [llama.cpp](https://github.com/ggerganov/llama.cpp). At present, the embeddings
llama.cpp generates has to be with a supported model and the Llama models generate embeddings the same size as their native context (e.g. 4096 dimensional arrays for llama2 derived models) which are unwieldy.


## Known Issues

* The whole context is processed with every inference currently.
* The log file format is not stable yet, though no compatibility breaks are anticipated at this time.
* Quote color highlighting only works for ascii quotes and not any UTF quote marks. Also, it doesn't 
  do syntax highlighting for emotes delimited by asterisks.
* Other participants currently don't have syntax highlighting until multi-chat mode is enabled.
* No way to cancel text generation currently.
* Mac and Windows builds currently do not support hardware accellerated sentence_similarity 
  testing with vector embeddings.
* Error messages will corrupt the output unless stderr is redirected to another terminal or file. Eventually
  this will be handled through message boxes instead.
* The new rust-llama.cpp backend implementation will sometimes cause a crash on a bad UTF-8 boundary,
  and this will be corrected soon.
* The memory feature needs to be documented once its feature set stabilizes.


## Caveats

It's still early days for the development of this app. The editing ability is pretty barebones and,
in general, I've mostly used it from a pretty powerful PC with a 4090, so it isn't as optimized
as it could be for a workfloww on a lower end device.


## License

This software is released under the AGPL v3 terms:

Copyright (C) 2023  Timothy Bogdala

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
