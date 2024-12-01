from dataclasses import dataclass
from typing import Any, Dict, List, Union


class GenerationTemplate:
    """
    Takes a prompt template and provides methods for filling in blanks.
    The format is as follows:
    [APE] is where text will be generated by the LLM.
    [full_DEMO] is where the full demo will be inserted.
    [INPUT] is where the input to the first demo will be inserted.
    [OUTPUT] is where the output from the first demo will be inserted.
    """

    def __init__(self, template: str):
        self.template = template
        # Check that the template is valid
        # There should be exactly one [APE] token
        assert self.template.count('[APE]') == 1

    def fill(self, full_demo='', input='', output=''):
        """
        Fills in the template with the given values.
        """
        return self.template.replace('[full_DEMO]', full_demo).replace(
            '[INPUT]', input).replace('[OUTPUT]', output)


class BackwardGenTemplate:
    """
    Takes a prompt template and provides methods for filling in blanks.
    The format is as follows:
    [INSTURCT] is the current instruct.
    [full_SUC_DEMO] is where the full sucess demo will be inserted.
    [full_FAIL_DEMO] is where the full sucess demo will be inserted.
    [MSG] is the additional message for backward.
    [APE] is where text will be generated by the LLM.
    """

    def __init__(self, template: str):
        self.template = template
        # Check that the template is valid
        # There should be exactly one [APE] token
        assert self.template.count('[APE]') == 1

    def fill(self, instruct, full_suc_demo, full_fail_demo, message):
        """
        Fills in the template with the given values.
        """
        text = self.template.replace('[INSTRUCT]', instruct).replace('[full_SUC_DEMO]', full_suc_demo) \
            .replace('[full_FAIL_DEMO]', full_fail_demo).replace('[MSG]', message)
        return text


class EvalTemplate:
    """
    Takes a prompt template and provides methods for filling in blanks.
    The format is as follows:
    [PROMPT] is where the prompt will be inserted.
    [full_DEMO] is where the full demo will be inserted.
    [INPUT] is where the input to the first demo will be inserted.
    [OUTPUT] is where the output from the first demo will be inserted.
    """

    def __init__(self, template):
        self.template = template
        self.prompt = ''

    def fill(self, prompt=None, full_demo='', input='', output=''):
    #def fill(self, prompt: str, full_demo: str, input: Any, output: str) -> str: #CHANGE HERE
        """
        Fills in the template with the given values.
        """

        # CHANGE HERE !
        #print(f"Template: {self.template}")
        #print(f"[PROMPT]: {prompt} (type: {type(prompt)})")
        #print(f"[full_DEMO]: {full_demo} (type: {type(full_demo)})")
        #print(f"[INPUT]: {input} (type: {type(input)})")
        #print(f"[OUTPUT]: {output} (type: {type(output)})")

        if prompt is None:
            prompt = self.prompt
        #return self.template.replace('[PROMPT]', prompt).replace('[full_DEMO]', full_demo).replace('[INPUT]', input).replace('[OUTPUT]', output)
        return self.template.replace('[PROMPT]', str(prompt)).replace('[full_DEMO]', str(full_demo)).replace('[INPUT]', str(input)).replace('[OUTPUT]', str(output)).rstrip()

    def convert_to_generation_template(self):
        """
        Converts the evaluation template to a generation template.
        """
        return GenerationTemplate(self.template.replace('[PROMPT]', '[APE]'))

    input_key = 'sentence'

    def encode(self, prompt, full_demo, input_dict):
        input = input_dict[self.input_key]
        return self.fill(prompt, full_demo, input, '').rstrip()


class DemosTemplate:
    """
    Takes a template for the full demo and provides methods for filling in blanks.
    The format is as follows:
    [INPUT], [OUTPUT]

    """

    def __init__(self, template, delimiter='\n\n'):
        self.template = template
        self.delimiter = delimiter

    def fill(self, data):
        """
        Fills in the template with the given values. Data is a tuple of lists.
        """
        demos = ''
        for i, (input_, output_) in enumerate(zip(*data)):
            demos += self.template.replace('[INPUT]', str(input_)).replace(
                '[OUTPUT]', str(output_))

            if i != len(data[0]) - 1:
                demos += self.delimiter

        return demos


class BwdDemosTemplate:
    """
    Takes a template for the full demo and provides methods for filling in blanks.
    The format is as follows:
    [INPUT], [OUTPUT], [TARGET]

    """

    def __init__(self, template, delimiter='\n\n'):
        self.template = template
        self.delimiter = delimiter

    def fill(self, bwd_infos):
        """
        Fills in the template with the given values. Data is a tuple of lists.
        """
        demos = ''
        for i, bwd_info in enumerate(bwd_infos):
            demos += self.template.replace('[INPUT]', bwd_info['input']).replace(
                '[OUTPUT]', bwd_info['output']).replace('[TARGET]', bwd_info['target'])

            if i != len(bwd_infos) - 1:
                demos += self.delimiter

        return demos


class SST2EvalTemplate(EvalTemplate):
    init_instruct = 'Classify the input text as positive or negative.'
    input_key = 'sentence'

    def verbalize(self, prompt, full_demo, input_dict):
        input = input_dict[self.input_key]
        output = input_dict['label_word']
        return self.fill(prompt, full_demo, input, output).rstrip()

    @classmethod
    def get_zeroshot_template(cls):
        return cls(("[INPUT]. It was [OUTPUT]"))


class TrecEvalTemplate(EvalTemplate):
    init_instruct = "Read the following question, then choose whether it is about a description, entity, expression, human, location or number."
    # init_instruct = "Read the following question, then choose whether it is about a abbreviation, entity, description, human, location or number."
    input_key = 'sentence'

    def verbalize(self, prompt, full_demo, input_dict):
        input = input_dict[self.input_key]
        output = input_dict['label_word']
        return self.fill(prompt, full_demo, input, output).rstrip()
    
    @classmethod
    def get_zeroshot_template(cls):
        return cls("[INPUT]  It was about [OUTPUT]")


class MpqaEvalTemplate(EvalTemplate):
    init_instruct = "Read the following review, then choose whether it is negative or positive."
    input_key = 'sentence'

    def verbalize(self, prompt, full_demo, input_dict):
        input = input_dict[self.input_key]
        output = input_dict['label_word']
        return self.fill(prompt, full_demo, input, output).rstrip()
    
    @classmethod
    def get_zeroshot_template(cls):
        return cls("[INPUT]\nThe answer is [OUTPUT]")

class DisasterEvalTemplate(EvalTemplate):
    init_instruct = 'Read the following sentence, then choose whether it is relevant to a disaster.'
    input_key = 'sentence'

    def verbalize(self, prompt, full_demo, input_dict):
        input = input_dict[self.input_key]
        output = input_dict['label_word']
        return self.fill(prompt, full_demo, input, output).rstrip()

    @classmethod
    def get_zeroshot_template(cls):
        return cls(("[INPUT]. Is it relevant to a disaster? [OUTPUT]"))



# ###### functions for get templates #####

def get_template_class(data_name):
    if data_name in ['sst2', 'sst2_priv']:
        temp_class = SST2EvalTemplate
    elif data_name in ['trec', 'trec_priv']:
        temp_class = TrecEvalTemplate
    elif data_name in ['mpqa', 'mpqa_priv']:
        temp_class = MpqaEvalTemplate
    elif data_name in ['disaster', 'disaster_priv']:
        temp_class = DisasterEvalTemplate
    else:
        raise NotImplementedError(f"Unknown data: {data_name}")
    return temp_class

def get_zeroshot_template(data_name):
    TempClass = get_template_class(data_name)
    return TempClass.get_zeroshot_template()

def get_eval_template(instruct_model, data_name, instruct_type='default', add_item_name=True):
    instruct_model = instruct_model.lower()
    TempClass = get_template_class(data_name)

    if instruct_type == 'default':
        if 'vicuna' in instruct_model or instruct_model == 'openai' or 'llama-2' in instruct_model \
            or 'llama2' in instruct_model or 'mistral' in instruct_model:
            instruct_type = 'vicuna'
        elif 'llama3' in instruct_model or 'llama-3' in instruct_model:
            # instruct_type = 'llama-3'  # not work well.
            instruct_type = 'vicuna'
        print(f"Choose instruct default template ({instruct_type}) for model.")
    
    if instruct_type == "vicuna":
        if add_item_name:
            eval_template = TempClass('Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOutput: [OUTPUT]')
        else:
            eval_template = TempClass('[PROMPT]\n\n[INPUT]\n\nOutput: [OUTPUT]')
    elif instruct_type == 'llama-2':
        if add_item_name:
            eval_template = TempClass('[INST] [PROMPT] [/INST]\n\nInput: [INPUT]\n\nOutput: [OUTPUT]')
        else:
            eval_template = TempClass('[PROMPT]\n\n[INPUT]\n\nOutput: [OUTPUT]')
    elif instruct_type == 'llama-3':
        if add_item_name:
            eval_template = TempClass('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|begin_of_text|> [PROMPT] <|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>\n\n[INPUT]\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>[OUTPUT]')
        else:
            eval_template = TempClass('[PROMPT]\n\n[INPUT]\n\nOutput: [OUTPUT]')
    else:
        raise NotImplementedError(f"Unknown instruct_type: {instruct_type}")
    return instruct_type, eval_template, TempClass.init_instruct


def get_bwd_template(instruct_type, privacy_instruct=-1):
    meta_instruct = "A student is completing a task that requires producing a text output from a text input. The student receives an instruction that describes how to produce the output given each input. The student has made some errors. Your task is to improve the instruction such that the student can fix the errors."
    content = """This was the instruction.

Instruction: [INSTRUCT]

# Student successes
[full_SUC_DEMO]

# Student errors
[full_FAIL_DEMO]

Improve the instruction to fix the student errors. [MSG]
Improved Instruction: [APE]"""

    if privacy_instruct == 1:
        meta_instruct = meta_instruct + " Do not provide examples in the prompt."
    elif privacy_instruct == 2:
        meta_instruct = meta_instruct + " Do not use existing samples but create dummy samples as examples in the prompt."

    if 'mpt' == instruct_type:
        sys_intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        raise NotImplementedError(f"instruct_type: {instruct_type}")
    elif instruct_type == 'vicuna':
        meta_template = BackwardGenTemplate(f"""{meta_instruct}\n{content}""")
    elif instruct_type == 'llama-2':
        meta_template = BackwardGenTemplate(f"""[INST] {meta_instruct} [/INST]\n{content}""")
    else:
        raise NotImplementedError(f"instruct_type: {instruct_type}")
    suc_demo_template = BwdDemosTemplate('Input: [INPUT]\nCorrect Output: [OUTPUT]')
    fail_demo_template = BwdDemosTemplate('Input: [INPUT]\nStudent Output: [OUTPUT]\nCorrect Ouput: [TARGET]')
    return meta_template, suc_demo_template, fail_demo_template


@dataclass
class DataCollatorWithOptAndTemplate:
    """
    Collator for training

    example should include keys:
        - SST2: [{'sentence': 'xxxx', 'label': 1}]
    """
    instruct: str
    label_words: List[List[str]]
    eval_template: EvalTemplate

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(examples[0], list):
            examples = [{'sentence': ex[0], 'label': ex[1]} for ex in examples]
        #change above
        batch = {}
        batch['text'] = []
        batch['candidate_targets'] = []
        batch['candidate_texts'] = []
        batch['labels'] = []
        for example in examples:
            text = self.eval_template.encode(prompt=self.instruct, full_demo='', input_dict=example)
            batch['text'].append(text)
            batch['candidate_targets'].append(self.label_words)

            candidate_texts = []
            for label_word in self.label_words:
                example['label_word'] = label_word
                cand = self.eval_template.verbalize(prompt=self.instruct, full_demo='', input_dict=example)
                candidate_texts.append(cand)
            batch['candidate_texts'].append(candidate_texts)
            batch['labels'].append(example['label'])
        return batch
