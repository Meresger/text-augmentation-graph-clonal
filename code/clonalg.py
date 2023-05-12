import AMR_functions as process
import random

# Instantiate objects
An_amr = process.AMRModels()
Mutate = process.AMRMutation()
processText = process.CleanText()

# Define available augmentation methods
augmentation_methods = [Mutate.concept_mutation, Mutate.mutation_rel, Mutate.mutation_top]

def process_sentence(sentence):
    """
    Process a single sentence by cleaning it, generating the AMR graph, and applying mutations for augmentation.
    Returns the augmented sentence.
    """
    clean_sentence = processText.cleanalltext(sentence)
    amr_graph = An_amr.get_single_amr(clean_sentence)
    augmented_graph = Mutate.concept_mutation(amr_graph)
    augmented_graph = Mutate.constant_mutation(augmented_graph)
    augmented_sentence = An_amr.get_text(augmented_graph)
    return augmented_sentence


def clonal_selection_augment(text, n_clones, fitness_fn, augmentation_methods):
    """
    Perform text augmentation using the Clonal Selection Algorithm (CLONALG).
    Generates clones of the text by applying random mutations and selects the best clones based on a fitness function.
    Returns the best mutated text.
    """
    clones = []
    for i in range(n_clones):
        augmented_text = random.choice(augmentation_methods)(text)
        fitness = fitness_fn(augmented_text, text)
        clones.append((augmented_text, fitness))

    # Sort the clones by fitness
    clones = sorted(clones, key=lambda x: x[1], reverse=True)

    # Generate new clones by mutating the selected ones
    new_clones = []
    for selected_clone, fitness in clones:
        mutated_clone = random.choice(augmentation_methods)(selected_clone)
        fitness = fitness_fn(mutated_clone, selected_clone)
        new_clones.append((mutated_clone, fitness))

    # Sort the new clones by fitness
    new_clones = sorted(new_clones, key=lambda x: x[1], reverse=True)

    return new_clones[0][0]


def clonal_selection_augment_iterations(text, n_clones, fitness_fn, augmentation_methods, iterations):
    """
    Perform multiple iterations of text augmentation using the Clonal Selection Algorithm (CLONALG).
    Generates clones of the text by applying random mutations and selects the best clones based on a fitness function.
    Returns the best mutated text after the specified number of iterations.
    """
    clones = []
    for i in range(n_clones):
        augmented_text = random.choice(augmentation_methods)(text)
        fitness = fitness_fn(augmented_text, text)
        clones.append((augmented_text, fitness))

    for iteration in range(iterations):
        # Sort the clones by fitness
        clones = sorted(clones, key=lambda x: x[1], reverse=True)

        # Select the best clones for further use
        selected_clones = clones[:int(n_clones / 2)]

        # Generate new clones by mutating the selected ones
        new_clones = []
        for selected_clone, fitness in selected_clones:
            mutated_clone = random.choice(augmentation_methods)(selected_clone)
            fitness = fitness_fn(mutated_clone, selected_clone)
            new_clones.append((mutated_clone, fitness))

        # Sort the new clones by fitness
        new_clones = sorted(new_clones, key=lambda x: x[1], reverse=True)
        clones = new_clones

    return clones

def get_clones_text_iter(x):
    """
    Apply clonal selection augmentation to a given text.
    Returns a list of augmented texts.
    """
    joint = []
    the_amr = An_amr.get_single_amr(x)
    new_clones = clonal_selection_augment_iterations(the_amr, 10, process.compare, augmentation_methods, 5)
    for i in new_clones:
        joint.append(An_amr.get_text(i[0]))
    return joint


def get_clone_text_no_iter(x):
    """
    Apply clonal selection augmentation to a given text.
    Returns a single augmented text.
    """
    the_amr = An_amr.get_single_amr(x)
    new_clones = clonal_selection_augment(the_amr, 10, process.compare, augmentation_methods)
    return An_amr.get_text(new_clones)


def get_clones_Large_text(x):
    """
    Apply clonal selection augmentation to a large text.
    Returns a string of augmented text.
    """
    joint = []
    snts = list(process.nlp(x).sents)
    for i in snts:
        try:
            joint.append(get_clone_text_no_iter(processText.cleanalltext(i)))
        except TypeError:
            print(i)
    return " ".join(joint)

#

