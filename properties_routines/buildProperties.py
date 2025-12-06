import re
from pathlib import Path

def build_masterfile():
    '''
    Creates a single file with all formulas phi, and the modsat set phi in a sketch format to build
    properties to use in solvers
    Note: this code should only be used when the modsat representation has modsat sets phi equal for every McNaughton function
    '''

    folder_list = ['./limodsat/nn_0', './limodsat/nn_1']
    j = -1
    for folder in folder_list:
        j+=1
        path = Path(folder) # folder with all neural networks modsat representations in Lukasiewicz \infty
        output_filename = f"nn_{j}_master.limodsat"
        string_key1 = "-= Formula phi =-"
        string_key2 = "f:"
        string_key3 = "-= MODSAT Set Phi =-"

        all_files = [
            fil for fil in path.glob('*.limodsat')
        ]

        total_files = len(all_files)

        with open("./properties_routines/"+output_filename, 'w', encoding='utf-8') as output_file:
            for i, file_path in enumerate(all_files):
                try:
                    content = file_path.read_text(encoding='utf-8')

                    content = content.replace(string_key1, string_key2) # replaces -= Formula phi =- for "f:" 
                    if string_key3 in content:
                        if i == total_files -1: # if its iterating over last file
                            content = content.replace(string_key3, "") # replaces -= MODSAT Set Phi =- for empty string 
                        else:
                            content = content.split(string_key3)[0] #gets only formula phi of each file if its not last file
                    output_file.write(content)
                except Exception as e:
                    print("Error in {file_path.name}: {e}")

def neuron_is_outputed(total_neurons, masterfile_name, nn_x):
    root_folder = Path('./properties_routines')
    master_path = root_folder / masterfile_name
    
    max_var = calculates_max_var(masterfile_name) + 1
    
    # split('f:') creates a list where each f: is a text block
    content_master = master_path.read_text(encoding='utf-8')
    blocks = content_master.split('f:')
    
    if not blocks[0].strip():
        blocks.pop(0)

    for i in range(total_neurons):
        new_file_name = f"{i}geqall.limodsat"
        new_path = Path(f'./properties/{nn_x}') / new_file_name
        
        with open(new_path, 'w', encoding='utf-8') as f_out:
            f_out.write("Sat\n\n")            
            for k, bloco in enumerate(blocks):
                cleaned_block = bloco.strip()
                if not cleaned_block:
                    continue 

                f_out.write("f:\n")

                if k < total_neurons:
                    partes = cleaned_block.split('::')
                    original_numbers = partes[-1].strip()
                    
                    logic_type = "Equivalence" if k == i else "Implication"
                    
                    f_out.write(f"Unit 1 :: Clause      :: {original_numbers}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {max_var}\n")
                    f_out.write(f"Unit 3 :: {logic_type} :: 1 2\n")
                
                else:
                    # copies the modsat set phi
                    f_out.write(cleaned_block + "\n")
                
                f_out.write("\n")
        
        print(f"Created: {new_file_name}")


def calculates_max_var(masterfile_name):
    file_path = Path('./properties_routines/'+masterfile_name)
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # re.findall(r'\d+', content): search all digits sequence in the text
        all_numbers = [int(num) for num in re.findall(r'\d+', content)]
        
        if all_numbers:
            return max(all_numbers)
        else:
            return 0
            
    except FileNotFoundError:
        print(f"Error: File '{masterfile_name}' was not found in {file_path.parent}")
        return None


def neuron_is_outputed_with_certainty(total_neurons, masterfile_name, nn_x):
    root_folder = Path('./properties_routines')
    master_path = root_folder / masterfile_name
    
    raw_max = calculates_max_var(masterfile_name)
    
    var_z = raw_max + total_neurons + 1
    
    content_master = master_path.read_text(encoding='utf-8')
    blocks = content_master.split('f:')
    
    if not blocks[0].strip():
        blocks.pop(0)

    # Loop i: Create one file for each neuron acting as the "Max" candidate
    for i in range(total_neurons):
        new_file_name = f"{i}geqall_certain.limodsat"
        new_path = Path(f'./properties/{nn_x}') / new_file_name
        
        # Ensure directory exists (optional, but good practice)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        with open(new_path, 'w', encoding='utf-8') as f_out:
            f_out.write("Sat\n\n")
            
            # --- PART 1: Iterate through blocks (Main Body) ---
            for k, block in enumerate(blocks):
                cleaned_block = block.strip()
                if not cleaned_block:
                    continue

                f_out.write("f:\n")

                if k < total_neurons:
                    # Logic: Each neuron k gets a unique variable (raw_max + k + 1)
                    # Unlike the previous function, here everyone is Equivalence 1 2
                    
                    parts = cleaned_block.split('::')
                    original_numbers = parts[-1].strip()
                    
                    # Variable specific to this block's neuron
                    current_neuron_var = raw_max + k + 1
                    
                    f_out.write(f"Unit 1 :: Clause      :: {original_numbers}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {current_neuron_var}\n")
                    f_out.write(f"Unit 3 :: Equivalence :: 1 2\n")
                
                else:
                    # Copies extra blocks (like set phi)
                    f_out.write(cleaned_block + "\n")
                
                f_out.write("\n")

            # --- PART 2: Footer A (Transition/Negation with Z) ---
            str_z_repeated_9 = " ".join([str(var_z)] * 9)
            
            f_out.write("f:\n")
            f_out.write(f"Unit 1 :: Clause      :: {var_z}\n")
            f_out.write(f"Unit 2 :: Clause      :: {str_z_repeated_9}\n")
            f_out.write(f"Unit 3 :: Negation    :: 2\n")
            f_out.write(f"Unit 4 :: Equivalence :: 1 3\n\n")

            # --- PART 3: Footer B (Max Verification Logic) ---
            # This part changes dynamically depending on 'i' (the current file's max candidate)
            f_out.write("f:\n")
            
            current_unit = 1
            unit_indices = []

            # Step 3.1: Define Clauses for ALL neurons
            for k in range(total_neurons):
                val = raw_max + k + 1
                f_out.write(f"Unit {current_unit} :: Clause      :: {val}\n")
                unit_indices.append(current_unit) # Store index for later reference
                current_unit += 1
            
            # Identify the unit index of our candidate 'i'
            target_unit_idx = unit_indices[i]

            # Step 3.2: Implications
            # We want to prove (Others -> Target). 
            # So for every k != i, we generate: Implication :: k_idx target_idx
            for k in range(total_neurons):
                if k == i:
                    continue # Don't imply itself
                
                source_idx = unit_indices[k]
                f_out.write(f"Unit {current_unit} :: Implication :: {source_idx} {target_unit_idx}\n")
                current_unit += 1

            # Step 3.3: Maximum List
            # The list contains all units EXCEPT the target one
            others_indices = [str(idx) for k, idx in enumerate(unit_indices) if k != i]
            str_others = " ".join(others_indices)
            
            idx_max_unit = current_unit
            f_out.write(f"Unit {current_unit} :: Maximum     :: {str_others}\n")
            current_unit += 1

            # Step 3.4: Clause Z repeated and Final Implication
            str_z_repeated_5 = " ".join([str(var_z)] * 5)
            idx_clause_z = current_unit
            
            f_out.write(f"Unit {current_unit} :: Clause      :: {str_z_repeated_5}\n")
            current_unit += 1
            
            f_out.write(f"Unit {current_unit} :: Implication :: {idx_clause_z} {idx_max_unit}\n")

        print(f"Created: {new_file_name}")


def get_max_var_from_text(content):
    """Helper to extract the maximum integer from a text string."""
    all_numbers = [int(num) for num in re.findall(r'\d+', content)]
    return max(all_numbers) if all_numbers else 0

def apply_robustness_transformation(a, b):
    # Ensure b is at least 1 to avoid negative repetition counts
    if b < 1:
        raise ValueError("Parameter 'b' must be greater than or equal to 1.")

    folder_list = ['./limodsat/nn_0', './limodsat/nn_1']
    j = -1
    
    for folder_str in folder_list:
        folder_path = Path(folder_str)
        j += 1

        # FIX 1: Wrap the string in Path() so the '/' operator works later
        output_folder = Path(f'./properties/nn_{j}/')
        
        # FIX 2: Ensure this directory exists, otherwise open() will fail
        output_folder.mkdir(parents=True, exist_ok=True)

        all_files = [
            fil for fil in folder_path.glob('*.limodsat')
        ]

        # Iterate over all files in the folder (ignoring subdirectories)
        for i, file_path in enumerate(all_files):
            if file_path.is_dir() or file_path.name.startswith('.'): 
                continue 
            
            try:
                # 1. Read Content & Calculate Max Var
                content = file_path.read_text(encoding='utf-8')
                
                # We calculate max_var from the raw content before any modification
                raw_max = get_max_var_from_text(content)
                new_var = raw_max + 1
                
                # 2. String Replacements
                # Standardize headers to 'f:'
                content = content.replace("-= Formula phi =-", "f:")
                content = content.replace("-= MODSAT Set Phi =-", "f:")
                
                # 3. Block Parsing
                blocks = content.split('f:')
                if not blocks[0].strip():
                    blocks.pop(0) # Remove empty leading block
                
                # Define output filename
                # Recommendation: Added .limodsat extension so the file is usable
                new_filename = f"{i}geq{a}_{b}.limodsat"
                
                # This works now because output_folder is a Path object
                output_path = output_folder / new_filename
                
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write("Sat\n\n")
                    
                    # --- BLOCK 0: Modify the original Formula Phi ---
                    if blocks:
                        first_block = blocks[0].strip()
                        
                        # Extract the original Unit 1 line
                        unit1_line = ""
                        for line in first_block.splitlines():
                            if "Unit 1" in line:
                                unit1_line = line.strip()
                                break
                        
                        if unit1_line:
                            # Generate the string with 'new_var' repeated 'a' times
                            str_repeated_a = " ".join([str(new_var)] * a)
                            
                            f_out.write("f:\n")
                            f_out.write(f"{unit1_line}\n")
                            f_out.write(f"Unit 2 :: Clause      :: {str_repeated_a}\n")
                            f_out.write(f"Unit 3 :: Implication :: 2 1\n\n")
                    
                    # --- MIDDLE BLOCKS: Copy the rest (old Set Phi) ---
                    # Iterate from the second block onwards
                    for block in blocks[1:]:
                        clean_block = block.strip()
                        if clean_block:
                            f_out.write("f:\n")
                            f_out.write(clean_block + "\n\n")
                            
                    # --- FINAL BLOCK: Add the new Robustness Logic ---
                    # Unit 2 has 'new_var' repeated (b-1) times
                    count_b = b - 1
                    str_repeated_b = " ".join([str(new_var)] * count_b)
                    
                    f_out.write("f:\n")
                    f_out.write(f"Unit 1 :: Clause      :: {new_var}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {str_repeated_b}\n")
                    f_out.write(f"Unit 3 :: Negation    :: 2\n")
                    f_out.write(f"Unit 4 :: Equivalence :: 1 3\n")

                print(f"Created: {output_path}")

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")


build_masterfile()
neuron_is_outputed(15, "nn_0_master.limodsat", "nn_0")
neuron_is_outputed(15, "nn_1_master.limodsat", "nn_1")
neuron_is_outputed_with_certainty(15, "nn_0_master.limodsat", "nn_0")
neuron_is_outputed_with_certainty(15, "nn_1_master.limodsat", "nn_1")
apply_robustness_transformation(7, 10)