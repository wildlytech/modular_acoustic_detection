import argparse
import json
import requests
import xmltodict

DOMAIN = "https://www.worldbirdnames.org/"
URL = DOMAIN + "master_ioc-names_xml.xml"
KEYTAGS = ['order', 'family', 'genus', 'species', 'subspecies']

NEXT_TAXONOMY = {'bird':'order',
                 'order':'family',
                 'family':'genus',
                 'genus':'species',
                 'species':'subspecies'}

# Nodes for the ontology json file
ONTOLOGY_EXTENSION_NODES = []

def parse_taxonomy(xml_dict, id_prefix, key):
    """
    Recursively perform depth-first search on taxonomy tree and
    flatten result for ontology extension.
    """

    next_key = NEXT_TAXONOMY.get(key)
    ontology_id = id_prefix

    latin_name = xml_dict.get('latin_name')
    if latin_name is None:
        # The only time this case should occur is for the highest node in tree
        # This node (type "bird") should not get added to the ontology
        node_ontology_entry = {"child_ids":[]}
    else:
        latin_name = latin_name.lower().replace("-", " ")
        name = latin_name

        if key == 'species':
            id_prefix_arr_split = id_prefix.split("_")

            # Get the genus name from the id prefix
            assert(id_prefix_arr_split[-2] == 'genus')
            genus = id_prefix_arr_split[-1]

            # If the genus isn't already in the name, add it
            # If the genus name is the same as the species name,
            # then repeat the name twice
            if (not name.startswith(genus)) or (genus == name):
                name = id_prefix_arr_split[-1] + " " + name
        elif key == 'subspecies':
            id_prefix_arr_split = id_prefix.split("_")

            # Get the species name from the id prefix
            assert((id_prefix_arr_split[-2] == 'species') and \
                   (id_prefix_arr_split[-4] == 'genus'))
            genus = id_prefix_arr_split[-3]
            species = id_prefix_arr_split[-1]

            # If the species isn't already in the name, add it
            # If the species name is the same as the subspecies name,
            # then repeat the name twice
            if (not name.startswith(species)) or (species == name):
                name = species + " " + name

            # If the genus isn't already in the name, add it
            # If the genus name is the same as the species AND subspecies name,
            # then repeat the name three times
            if (not name.startswith(genus)) or \
               ((genus == species) and (genus == latin_name)):
                name = genus + " " + name

        name = name.replace('-', ' ')

        english_name = xml_dict.get('english_name')
        if english_name:
            english_name = english_name.lower()
        note = xml_dict.get('note')

        ontology_id = id_prefix+"_{0}_{1}".format(key, latin_name.replace(' ', '-'))

        description = "Bird {0} {1}(Common name:{2}) (Note:{3})".format( \
                            key, name, english_name, note)
        url = xml_dict.get('url')
        if url:
            url = DOMAIN+url

        # Some information is not necessary for the ontology
        # (e.g. authority, common name, breeding regions and subregions)
        # But they are added for reference. The training flow will never
        # reference those keywords
        node_ontology_entry = {"id": ontology_id,
                               "name": name,
                               "common_name": english_name,
                               "extinct": xml_dict.get("@extinct"),
                               "authority": xml_dict.get("authority"),
                               "breeding_regions": xml_dict.get("breeding_regions"),
                               "breeding_subregions": xml_dict.get("breeding_subregions"),
                               "description": description,
                               "citation_uri": url,
                               "audioset_parent_id": "/m/015p6" if key == 'order' else None,
                               "child_ids":[]}

        # Add node to ontology
        ONTOLOGY_EXTENSION_NODES.append(node_ontology_entry)

    # Parse all the immediate child nodes
    elements = xml_dict.get(next_key)

    # If there is no child node, then this is a leaf node
    if elements is not None:
        # elements could be a single item or list
        elements = elements if type(elements) == list else [elements]

        for element in elements:
            child_id = parse_taxonomy(element, ontology_id, next_key)

            # Make sure to add ids of all the children to this node
            node_ontology_entry['child_ids'].append(child_id)

    return ontology_id

########################################################################
# Main Function
########################################################################
if __name__ == "__main__":

    DESCRIPTION = 'Scrape IOC World Bird Taxonomy and generate Ontology extension file'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-o', '--OUTPUT_FILE', action='store', \
                                    help='Path to save output ontology extension file',
                                    required=True)
    PARSED_ARGS = PARSER.parse_args()

    ONTOLOGY_EXT_FILE_PATH = PARSED_ARGS.OUTPUT_FILE

    print("Downloading IOC taxonomy data ...")
    RESPONSE = requests.get(URL, headers={'user-agent': 'my-app/0.0.1'})

    print("Converting taxonomy data to ontology extension format...")
    TAXONOMY_DICT = xmltodict.parse(RESPONSE.text)

    parse_taxonomy(TAXONOMY_DICT['ioclist']['list'], 'bird_ioc', 'bird')

    # Write out the ontology to json
    with open(ONTOLOGY_EXT_FILE_PATH, 'w') as file_obj:

        json_data = json.dumps(ONTOLOGY_EXTENSION_NODES,
                               indent=4, sort_keys=True)

        file_obj.write(json_data)

        print("Wrote JSON Data to", ONTOLOGY_EXT_FILE_PATH)
