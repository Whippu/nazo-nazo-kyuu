from collections import defaultdict
import math
import pdb
import os
import pickle
from itertools import product
import struct
from collections import defaultdict
from functools import partial

"""
# Some example code here to get you started!

decoder = FileDecoder(r'D:\Documents\Games\Game Files')
decoder.open('MSysMsg.dat')  # Sets the current active file in decoder and creates a PointerTableDecoder for the file
decoder.bytes_to_text(output_rpl=True)  # Outputs a .txt file with the translated text. If output_rpl=True, creates a copy ending in .rpl as well
# Here you would make your modifications to the .rpl file. When you are ready to convert it into bytes:
decoder.load_from_replacement_file()  # This will output a RECONSTRUCTED version of your file which you can drop into Tinke as a replacement

"""


"""
Underlying classes here!
"""
class BytesDecoder(object):
    """
    The main class which handles conversion of raw bytes to readable text. Pass in the raw bytes and a mapping
    table which you want to use to convert the text, then call bytes_to_text.
    """


    def __init__(self, raw_bytes, table):
        self.bytes = raw_bytes
        self.table = table

    def bytes_to_text(self, as_bytes=False):
        new_text_chars = []
        counter = 0
        text_len_counter = 0
        control_object = Chikuhouse(self.bytes, self.table)

        while counter < len(self.bytes):
            counter_increment = 1
            byte = self.bytes[counter]

            if as_bytes:
                text_to_append = f'[{hex_str(byte)}]'
            elif byte == Chikuhouse.CONTROL_CODE:
                text_to_append, counter_increment = control_object.process_control_code(counter)
            else:

                try:
                    orig_byte = byte
                    text_to_append = self.table
                    counter_increment = 0
                    while isinstance(text_to_append, dict):
                        byte = self.bytes[counter + counter_increment]
                        text_to_append = text_to_append[byte]
                        counter_increment += 1
                except (KeyError, IndexError):
                    counter_increment = 1
                    text_to_append = f'[{hex_str(orig_byte)}]'

            new_text_chars.append(text_to_append)
            counter += counter_increment
            text_len_counter += len(text_to_append)

        new_text = ''.join(new_text_chars)
        return new_text

class TextToBytesProcessor(object):
    """
    A class which takes the text and turns it back into bytes. Should be a 1 to 1 conversion from BytesDecoder.
    """

    LITERAL_BYTE_DELIMITERS = '[]'

    def __init__(self, text, table):
        assert isinstance(text, str)
        self.text = text
        self.table = table

    def convert_text_to_bytes(self):
        text = self.text
        reverse_table = make_reverse_table(self.table)
        control_object = Chikuhouse(self.text, self.table)

        corresponding_bytes = bytes()
        counter = 0

        while counter < len(self.text):
            char = self.text[counter]
            if char == self.LITERAL_BYTE_DELIMITERS[0]:
                ending_delimiter_location = text.index(self.LITERAL_BYTE_DELIMITERS[1], counter)
                raw_text_captured = text[counter+1:ending_delimiter_location]

                bytes_to_append = bytes([int(raw_text_captured, 16)])
                increment = len(raw_text_captured) + 2

            elif char == Chikuhouse.DELIMITERS[0]:
                bytes_to_append, increment = control_object.process_control_code(counter)
            elif char == '\n':
                increment = 1
                bytes_to_append = bytes()
            else:
                try:
                    bytes_to_append = reverse_table[char]
                    increment = 1
                except ValueError:
                    raise ValueError('Unknown character "{}" detected in text!')

            counter += increment
            corresponding_bytes += bytes_to_append

        return corresponding_bytes



class PointerTableDecoder(object):

    """
    Class which deals with managing files represented as pointer tables which point to other pieces of data. Can be
    used to output readable reversible text files and convert those text files back into bytes. Can also handle
    nested pointer tables.
    """

    DELIMITER = '---- POINTER TABLE: '
    ENDING_DELIMITER = '---- END POINTER TABLE: '
    POINTER_INDICATOR = '-- Pointer: '

    def __init__(self, raw_bytes, table, offset=0, metadata=None, include_endpoints = False, **kwargs):
        self.bytes = raw_bytes
        self.table = table
        self.offset = offset
        self.pointer_map = {}
        self.include_endpoints = include_endpoints

        if metadata is None:
            metadata = {}

        pointers = parse_pointer_table(raw_bytes[8 if include_endpoints else 0:])
        for start, end in zip(pointers, pointers[1:] + [None]):
            if include_endpoints:
                start += 8
                if end is not None:
                    end += 8

            decoder_class = metadata.get(start + offset, metadata.get('Default', BytesDecoder))
            kwargs = {}
            if issubclass(decoder_class, PointerTableDecoder):
                kwargs = dict(offset=start)

            self.pointer_map[start] = decoder_class(self.bytes[start:end], table, **kwargs)

    def bytes_to_text(self, as_bytes=False):
        text = self.DELIMITER + str(self.offset) + '\n'
        all_pointers = sorted(self.pointer_map.keys())

        for pointer in all_pointers:

            absolute_pointer = self.offset + pointer
            text += self.POINTER_INDICATOR + str(absolute_pointer) + '\n'
            text += self.pointer_map[pointer].bytes_to_text(as_bytes=as_bytes)
            text += '\n\n'

        text += self.ENDING_DELIMITER + str(self.offset) + '\n'
        return text

    def construct_new_table(self, replacement_data):

        # Replacement data
        pointer_info = self.parse_replacement_data(replacement_data)

        pointers = []
        all_pointers = sorted(pointer_info)

        byte_output = bytes()
        base_offset = len(all_pointers) * 4  # Implicit hardcode

        current_offset = 0
        for pointer in all_pointers:

            data_to_convert = pointer_info[pointer]
            corresponding_decoder = self.pointer_map[pointer - self.offset]

            if isinstance(corresponding_decoder, PointerTableDecoder):
                # Note! Implicit assumption here we only ever go 2 pointer tables deep. If need more, check how to
                # pass in metadata
                new_decoder = PointerTableDecoder(corresponding_decoder.bytes, self.table, corresponding_decoder.offset,
                                                  include_endpoints=corresponding_decoder.include_endpoints)
                bytes_to_append = new_decoder.construct_new_table(data_to_convert)
            else:
                bytes_to_append = TextToBytesProcessor(data_to_convert, table=self.table).convert_text_to_bytes()

            pointers.append(base_offset + current_offset)

            byte_output += bytes_to_append
            current_offset += len(bytes_to_append)

        new_bytes = reconstruct_pointer_table(pointers) + byte_output
        new_bytes += bytes([0]) * ((4 - len(new_bytes) % 4) % 4)

        if self.include_endpoints:
            new_bytes = struct.pack('<I', 8) + struct.pack('<I', len(new_bytes) + 8) + new_bytes

        return new_bytes

    @classmethod
    def parse_replacement_data(cls, replacement_data):

        # Replacement data is a text stream from the .rpl file which is output from bytes_to_text.
        if isinstance(replacement_data, dict):
            return replacement_data

        rez = {}

        if isinstance(replacement_data, str):
            all_lines = replacement_data.split('\n')
        else:
            all_lines = replacement_data

        current_line = 0
        current_pointer = None

        while current_line < len(all_lines):
            line = all_lines[current_line].strip('\n')
            if not line or line.startswith('#'):
                current_line += 1
                continue

            if line.startswith(cls.DELIMITER):
                argument = int(line.replace(cls.DELIMITER, ''))
                ending_line = cls.ENDING_DELIMITER + str(argument)
                ending_line_location = all_lines.index(ending_line)
                rez[argument] = cls.parse_replacement_data(all_lines[current_line + 1:ending_line_location])

                current_line = ending_line_location + 1

            elif line.startswith(cls.POINTER_INDICATOR):
                argument = int(line.replace(cls.POINTER_INDICATOR, ''))
                current_pointer = argument
                rez[current_pointer] = ''

                current_line += 1
            else:
                rez[current_pointer] += line
                current_line += 1

        if 0 in rez:
            return rez[0]
        return rez

class PointerTableDecoderWithEndpoints(PointerTableDecoder):
    def __init__(self, *args, **kwargs):
        kwargs['include_endpoints'] = True
        super(PointerTableDecoderWithEndpoints, self).__init__(*args, **kwargs)


class FileDecoder(object):

    DECODER_MAP = {
        'PointerTableEndpoints': PointerTableDecoderWithEndpoints,
        'PointerTableNoEndpoints': PointerTableDecoder,
        'Bytes': BytesDecoder,
    }

    def __init__(self, path):
        self.loader = Hifhif(path, load=True)
        self.table = self.loader.table
        self.decoders = {}
        self.current_file = None
        self.metadata = defaultdict(dict)
        self._load_metadata()

    def open(self, file_name):
        raw_bytes = self.loader.get_text(file_name)

        self.decoders[file_name] = PointerTableDecoder(raw_bytes, self.table, metadata=self.metadata[file_name])
        self.current_file = file_name

    def bytes_to_text(self, as_bytes=False, output_rpl = False):
        text = self.decoders[self.current_file].bytes_to_text(as_bytes=as_bytes)
        suffix = '{}.txt'.format('.bytes' if as_bytes else '')
        output_file_name = os.path.join(self.loader.path, self.loader.DATA_FOLDER, self.current_file + suffix)
        with open(output_file_name, 'w', encoding='utf-8') as fh:
            fh.write(text)

        if output_rpl:
            rpl_file_name = output_file_name.replace('.txt', '.rpl')
            with open(rpl_file_name, 'w', encoding='utf-8') as fh:
                fh.write(text)

        print(f'Output to: {output_file_name}')

    def load_from_replacement_file(self, file_name = None):
        file_name = file_name or self.current_file
        new_file_name = os.path.join(self.loader.path, self.loader.DATA_FOLDER, 'RECONSTRUCTED_' + self.current_file)
        with open(new_file_name, 'wb') as fh:
            fh.write(self.decoders[file_name].construct_new_table(self._load_replacement_data()))
        print(f'Output to: {new_file_name}')


    def _load_metadata(self):
        metadata_file = os.path.join(self.loader.path, self.loader.DATA_FOLDER, self.loader.METADATA_FILE)
        with open(metadata_file, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()

        current_file = None
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith('!'):
                current_file = line.split(':')[0][1:]
                if ':' in line:
                    decoder_class = line.split(':')[1]
                    self.metadata[current_file]['Default'] = self.DECODER_MAP[decoder_class]
                else:
                    self.metadata[current_file]['Default'] = BytesDecoder

                continue
            if ':' in line:
                pointer, decoder_class = line.split(':')
                pointer = int(pointer)
                self.metadata[current_file][pointer] = self.DECODER_MAP[decoder_class]

    def _load_replacement_data(self, file_name=None):
        if file_name is None:
            file_name = self.current_file

        rpl_file_name = os.path.join(self.loader.path, self.loader.DATA_FOLDER, file_name + '.rpl')
        try:
            with open(rpl_file_name, 'r', encoding='utf-8') as fh:
                text = fh.read()
        except FileNotFoundError:
            print('No replacement data found for file: ' + rpl_file_name)
            return {}

        return text

    def search_for_text(self, phrase):
        for file_name in self.loader.files:
            self.open(file_name)
            text = self.decoders[file_name].bytes_to_text()
            if phrase in text:
                print(f'Found {phrase} in {file_name}')

class Chikuhouse(object):
    """
    This object takes care of handling control code logic for Nazo Nazo Q
    """

    CONTROL_CODE = 255
    DELIMITERS = '{}'

    KNOWN_CONTROL_CODES = {

        # Control codes are defined with the following metadata:
        # Key: Output tag, search for ending tag (None if no need, else bytes for end), append newline

        (0,): ['NL', None, True],
        (1, 0): ['ResetText', None, True],
        (10, 0): ['EndDialogue', None, True],
        (11, 1): ['BoxStart', None, True],
        (17, 0): ['ClearText', None, False],
        (17, 1): ['UserInput', None, False],
        (35,): ['Red', (255, 32), False],
        (38,): ['Blue', (255, 32), False],
        (51,): ['BigText', None, False],
        (53,): ['CenterLine', None, False],
    }

    def __init__(self, text_or_bytes, table):
        """
        Chikuhouse requires the body of text you'd like to search through as well as the table mapping, since some
        control sequences return text that is translatable.
        :param text:
        """
        self.text_or_bytes = text_or_bytes
        if isinstance(self.text_or_bytes, bytes):
            self.mode = 'Forward'
        else:
            self.mode = 'Reverse'
        self.table = table
        self.counter = 0

    def process_control_code(self, counter):
        """
        Given some text and the position of the control code indicated, return the replacement text and the number
        of steps to advance the text counter.
        """
        if self.mode == 'Forward':
            return self._forward_process_control_code(counter)
        else:
            return self._reverse_control_code(counter)


    def _forward_process_control_code(self, counter):

        raw_bytes = self.text_or_bytes
        assert isinstance(raw_bytes, bytes)
        assert raw_bytes[counter] == self.CONTROL_CODE

        max_len = max([len(k) for k in self.KNOWN_CONTROL_CODES.keys()])

        for increment in range(1, max_len+1):
            start = counter + 1
            end = start + increment
            control_sequence = tuple(raw_bytes[start:end])
            try:
                tag_name, ending_tag, add_nl = self.KNOWN_CONTROL_CODES[control_sequence]
                if ending_tag is None:

                    return self.DELIMITERS[0] + tag_name + self.DELIMITERS[1] + ('\n' if add_nl else ''), 1 + increment
                else:
                    # Search for the ending tag
                    ending_bytes = bytes(ending_tag)
                    try:
                        ending_tag_position = raw_bytes.index(ending_bytes, counter)
                    except ValueError:
                        # print('Couldn\'t find ending tag for {}'.format(tag_name))
                        break

                    # Whatever we capture between the ending tags needs to be processed as well, so we feed it into a
                    # BytesDecoder and try again
                    captured_bytes = raw_bytes[end:ending_tag_position]
                    captured_text = BytesDecoder(captured_bytes, self.table).bytes_to_text()

                    starting_tag = self.DELIMITERS[0] + tag_name + self.DELIMITERS[1]
                    ending_tag = self.DELIMITERS[0] + '/' + tag_name + self.DELIMITERS[1]

                    num_bytes_processed = ending_tag_position + len(ending_bytes) - counter

                    return starting_tag + captured_text + ending_tag + ('\n' if add_nl else ''), num_bytes_processed

            except KeyError:
                continue

        # Unknown control sequence, just return the original control code
        return f'[{hex_str(255)}]', 1

    def _reverse_control_code(self, counter):
        text = self.text_or_bytes
        assert isinstance(text, str)
        assert text[counter] == self.DELIMITERS[0]

        swapped_table = {tag_name: [control_code, ending_bytes] for control_code, (tag_name, ending_bytes, _) in self.KNOWN_CONTROL_CODES.items()}
        try:
            opening_tag_ending_delimiter_loc = text.index(self.DELIMITERS[1], counter)
        except ValueError:
            raise ValueError('Found an opening bracket with no closing bracket!')
        tag_name = text[counter+1:opening_tag_ending_delimiter_loc]
        try:
            control_code, ending_bytes = swapped_table[tag_name]
            if ending_bytes is None:
                # Simply replace the corresponding tag with the corresponding bytes
                return bytes([255]) + bytes(control_code), len(tag_name) + 2
            else:

                ending_tag = self.DELIMITERS[0] + '/' + tag_name + self.DELIMITERS[1]
                ending_tag_location = text.index(ending_tag, counter)
                captured_text = text[opening_tag_ending_delimiter_loc+1:ending_tag_location]
                captured_bytes = TextToBytesProcessor(captured_text, self.table).convert_text_to_bytes()

                starting_bytes = bytes([255]) + bytes(control_code)
                ending_bytes = bytes(ending_bytes)
                num_chars_processed = (len(tag_name) + 2) + len(ending_tag) + len(captured_text)

                return starting_bytes + captured_bytes + ending_bytes, num_chars_processed
        except KeyError:
            raise ValueError('I found an unknown tag name "" in the file!'.format(tag_name))


def parse_pointer_table(raw_bytes):

    format_code = '<I'
    int_size = 4

    pointers = []

    current_loc = 0
    while not len(pointers) or current_loc < pointers[0]:
        new_pointer = struct.unpack(format_code, raw_bytes[current_loc:current_loc + int_size])[0]
        pointers.append(new_pointer)
        current_loc += int_size

    return pointers

def reconstruct_pointer_table(pointers, format_code='<I', offset=0):

    raw_bytes = bytes()
    for pointer in pointers:
        raw_bytes += struct.pack(format_code, pointer + offset)

    return raw_bytes


def make_reverse_table(table):

    # Returns a dictionary indexed by a text character, with the byte values to append
    rez = {}
    for key, val in table.items():

        if not isinstance(val, dict):
            rez[val] = bytes([key])
            continue

        subtable = make_reverse_table(table[key])
        for text_char, byte_seq in subtable.items():
            rez[text_char] = bytes([key]) + byte_seq

    return rez

def hex_str(n):
    return hex(n)[2:].upper()

# Semi deprecated! Most functionality is useless, I just need to move the code that loads metadata into another class...

class Hifhif(object):
    """
    A class used for finding patterns in raw bytes that correspond to human-readable text.
    Given a number of seed phrases, it will search through all the files and look for possible mappings of the raw
    bytes in the file to the seed phrases.es.
    """

    CACHE_FILE = 'map.tbl'
    SEED_FILE = '__seed.txt'
    METADATA_FILE = '__metadata.txt'
    DATA_FOLDER = 'data'

    CONTROL_CODE = 255

    def __init__(self, path, load=True):
        """
        The Hifhif class's only identifier is the location of all the files you wish to process.
        :param path: Absolute file path with the files to process
        :param load: If True, it will look for and attempt to load previously complete dictionaries.
        """

        self.path = path
        self.table = dict()
        self.files = [file_name for file_name in os.listdir(path) if file_name not in [self.CACHE_FILE, self.SEED_FILE,
                                                                                       self.DATA_FOLDER,
                                                                                       self.METADATA_FILE]]
        if load:
            self.load()

    def save(self):
        raise NotImplementedError('This needs to be fixed for the multi byte characters!')

        table_file_lines = []
        for byte_code in sorted(self.table.keys()):
            table_file_lines.append(f'{byte_code}={self.table[byte_code]}')

        with open(os.path.join(self.path, self.DATA_FOLDER, self.CACHE_FILE), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(table_file_lines))

    def load(self):
        try:
            with open(os.path.join(self.path, self.DATA_FOLDER, self.CACHE_FILE), encoding='utf-8') as fh:
                table_file_lines = fh.readlines()
        except FileNotFoundError:
            print('No cached results were located for this directory')
            return

        loaded_table = {}
        for line in table_file_lines:
            byte_str_sequence, char = line.strip('\n').split('=', maxsplit=1)[:2]
            byte_str_partition = [byte_str_sequence[2 * n:2 * (n + 1)] for n in range(len(byte_str_sequence) // 2)]
            byte_sequence = [int(byte_str, 16) for byte_str in byte_str_partition]

            current_dict_reference = loaded_table
            for seq, byte in enumerate(byte_sequence, start=1):
                if seq == len(byte_sequence):
                    current_dict_reference[byte] = char
                else:
                    if current_dict_reference.get(byte) is None:
                        current_dict_reference[byte] = {}
                    current_dict_reference = current_dict_reference[byte]

        self.table = loaded_table

    def update_seeds(self, seeds, overwrite=False):
        """
        Updates the seed list
        :param overwrite: If set to True, will erase the existing seed file
        :return: Nothing.
        """
        if isinstance(seeds, str):
            seeds = [seeds]

        if not overwrite:
            current_seeds = self.retrieve_seeds()
        else:
            current_seeds = []

        for seed in seeds:
            if seed not in current_seeds:
                current_seeds.append(seed)

        with open(os.path.join(self.path, self.DATA_FOLDER, self.SEED_FILE), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(current_seeds))

    def retrieve_seeds(self):
        try:
            with open(os.path.join(self.path, self.DATA_FOLDER, self.SEED_FILE), encoding='utf-8') as fh:
                raw_text = fh.read()
                seeds = [seed.strip().strip('\ufeff').replace('\u3000', ' ')
                         for seed in raw_text.split('\n')]
                seeds = [seed for seed in seeds if seed]
            return seeds

        except FileNotFoundError:
            print(f'Could not find a {self.SEED_FILE} file')
            return []

    def translate_text(self, file_name, replace_unknown=True, output=True):
        """
        With the existing dictionary mapping, translate all text that is currently available.
        :param file_name:
        :param chunk_size:
        :return:
        """

        raise NotImplementedError('Replaced, rip')

        text = self.get_text(file_name)
        table = self.table.copy()
        control_object = Chikuhouse(text, table)

        new_text_chars = []
        counter = 0
        while counter < len(text):
            counter_increment = 1
            byte = text[counter]
            if byte == Chikuhouse.CONTROL_CODE:

                text_to_append, counter_increment = control_object.process_control_code(counter)
                new_text_chars.append(text_to_append)
            else:
                new_text_chars.append(table.get(byte, f'[{byte}]' if replace_unknown else byte))

            counter += counter_increment

        new_text = ''.join(new_text_chars)
        if output:
            with open(os.path.join(self.path, self.DATA_FOLDER, file_name + '.txt'), 'w', encoding='utf-8') as fh:
                fh.write(new_text)

        return new_text

    def get_text(self, file_name):
        with open(os.path.join(self.path, file_name), 'rb') as fh:
            text = fh.read()
        return text

    def search_for_seeds(self, file_name, chunk_size=1):
        """
        The main function of this class. Given the seeds in the seed file, will look through a file for instances of
        this pattern. If it finds a single consistent mapping for all, it will update the master table.

        :param file_name: The file you wish to search through.
        :param chunk_size:
        :return: A summary of mappings for all files.
        """

        seeds = self.retrieve_seeds()
        if not len(seeds):
            raise ValueError('You haven\'t loaded any seeds to search for! Please add some using update_seeds.')

        text = self.get_text(file_name)
        existing_mapping = self.table.copy()

        all_potential_proposals = [existing_mapping]
        for seed in seeds:
            new_potential_proposals = []

            for potential_proposal in all_potential_proposals:
                proposals = self.locate_potential_patterns(seed, text, chunk_size=chunk_size,
                                                           validation_dict=potential_proposal)
                proposals = self.remove_duplicate_dicts(proposals)

                new_potential_proposals.extend(proposals)

            all_potential_proposals = new_potential_proposals

        if len(all_potential_proposals) == 1:
            print(f'Found consistent mapping for {file_name}!')
            self.table = all_potential_proposals[0]
        elif len(all_potential_proposals) > 1:
            print(f'Found multiple consistent mappings for {file_name}! Try adding more seeds.')
            pdb.set_trace()

    def merge_proposals(self, proposals_by_seed):
        """
        This function takes a list of potential mappings and attempts to create consistent merged mappings.
        :param proposals: A list of potential mappings corresponding to each seed.
        :return: A list of potential merged mappings.
        """

        merged_results = []

        for mapping_set in product(*proposals_by_seed):
            initial_mapping = {}
            for mapping in mapping_set:
                if self.dicts_are_consistent(initial_mapping, mapping, enforce_equality=False):
                    initial_mapping.update(mapping)
                else:
                    break
            else:
                merged_results.append(initial_mapping)
        return merged_results

    @staticmethod
    def dicts_are_consistent(dict1, dict2, enforce_equality=False):

        if enforce_equality:
            if set(dict1.keys()) != set(dict2.keys()):
                return False

        reverse_dict1 = {v: k for k, v in dict1.items()}
        reverse_dict2 = {v: k for k, v in dict2.items()}

        for d1, d2 in [[dict1, dict2], [reverse_dict1, reverse_dict2]]:
            intersecting_keys = set(d1.keys()).intersection(set(d2.keys()))
            for key in intersecting_keys:
                if d1[key] != d2[key]:
                    return False

        return True

    @staticmethod
    def remove_duplicate_dicts(dict_list):
        # Remove duplicate proposals
        final_list = []
        for proposal in dict_list:
            for valid_proposal in final_list:
                if Hifhif.dicts_are_consistent(proposal, valid_proposal, enforce_equality=True):
                    break
            else:
                final_list.append(proposal)
        return final_list

    @staticmethod
    def chunk_text(text, chunk_size):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    @staticmethod
    def locate_potential_patterns(pattern, text, chunk_size=1, validation_dict=None):
        """
        This function does the heavy lifting of identifying potential byte->pattern character mappings. Given a pattern
        and a body of text to search through, it will return a list of potential mappings that match.
        :param pattern: A string indicating the pattern you're looking for
        :param text: A string for the body of text you're searching for
        :param chunk_size: The size of the blocks which should be matched.
        :param validation_dict: A pre-existing dictionary which the proposed patterns must conform to. If a mapping
            disagrees with the input dictionary, the result is discarded.
        :return:
        """

        pattern_data = defaultdict(list)
        chunked_pattern = []
        paren_mode = False
        current_paren_chunk = ''
        for char in pattern:
            if not paren_mode:
                if char == '(':
                    paren_mode = True
                else:
                    chunked_pattern.append(char)
            else:
                if char == ')':
                    paren_mode = False
                    chunked_pattern.append(current_paren_chunk)
                    current_paren_chunk = ''
                else:
                    current_paren_chunk += char

        if paren_mode:
            raise ValueError('You have an unfinished paren in your pattern!')

        for i in range(len(chunked_pattern)):
            # We take note of the characters, but we also take note of special instances
            key = chunked_pattern[i]
            pattern_data[key].append(i)

        chunks = Hifhif.chunk_text(text, chunk_size)
        length = len(chunked_pattern)

        if validation_dict is None:
            validation_dict = {}

        existing_chars = set(validation_dict.values())
        possible_mappings = []

        for iteration in range(len(chunks)):
            start_pt = iteration
            end_pt = start_pt + length

            if end_pt > len(chunks):  # Lazy, I know...
                break

            chunks_to_search = chunks[start_pt:end_pt]

            claimed_chunks = set()
            proposed_mapping = dict()

            for char, char_locations in pattern_data.items():

                proposed_chunk = None
                for location in char_locations:

                    potential_chunk = chunks_to_search[location]  # Agreement means we check the rest of the locations
                    # If the character doesn't already have a mappi
                    if potential_chunk == proposed_chunk:
                        continue
                    elif proposed_chunk is None:  # For the initial loop
                        proposed_chunk = potential_chunk
                    else:  # The proposed chunks do not agree, this part of the text cannot be a match to our pattern
                        proposed_chunk = False
                        break
                if proposed_chunk is False:
                    break
                else:
                    # If another character has already mapped to this chunk, we fail and move on
                    if proposed_chunk in claimed_chunks:
                        break

                    # Chunk <-> character mappings must be one to one.
                    # If we detect that the chunk maps to a different character already, or if we detect that
                    # the character in question has a different chunk mapping to it, we discard this case.
                    validation_char = validation_dict.get(proposed_chunk)
                    if (validation_char is not None or char in existing_chars) and char != validation_char:
                        break

                    claimed_chunks.add(proposed_chunk)
                    proposed_mapping[proposed_chunk] = char
            else:
                # If all characters were successfully mapped to a chunk, we propose this mapping and move on.
                proposed_mapping.update(validation_dict)
                possible_mappings.append(proposed_mapping)

        return possible_mappings
