import requests

# Define the API URL with the specific RxCUI values
d1 = input("Drug Name 1")
d2 = input("Drug Name 2")
ra = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={d1}&search=0"
r1 = requests.get(ra)
if r1.status_code == 200:
    r1_data = r1.json()
else:
    print("Enter valid name")
rb = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={d2}&search=0"
r2 = requests.get(rb)
if r2.status_code == 200:
    r2_data = r2.json()
else:
    print("Enter valid name")
#print(r1_data)

api_url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={r1_data['idGroup']['rxnormId'][0]}+{r2_data['idGroup']['rxnormId'][0]}&source=DrungBank"

#api_url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis=2551+161&sources=DrugBank"


try:
    # Send a GET request to the API
    response = requests.get(api_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the response content as JSON
        response_data = response.json()
        print(response_data)
        
        if 'fullInteractionTypeGroup' in response_data:
            description = response_data["fullInteractionTypeGroup"][0]["fullInteractionType"][0]["interactionPair"][0]["description"]
            print(severity)
            print(description)
        
      
        """# Check for a "fullInteractionTypeGroup" in the JSON data
        if 'fullInteractionTypeGroup' in response_data:
            # Iterate through each interaction group
            for interaction_group in response_data['fullInteractionTypeGroup']:
                source_name = interaction_group['sourceName']
                source_disclaimer = interaction_group['sourceDisclaimer']

                # Iterate through each interaction type
                for interaction_type in interaction_group['fullInteractionType']:
                    comment = interaction_type['comment']
                    #severity = interaction_type['severity']

                    # Iterate through each interaction pair
                    for interaction_pair in interaction_type['interactionPair']:
                        # Extract information about the two interacting concepts
                        concept1 = interaction_pair['interactionConcept'][0]['minConceptItem']
                        concept2 = interaction_pair['interactionConcept'][1]['minConceptItem']

                        rxcui1 = concept1['rxcui']
                        name1 = concept1['name']
                        tty1 = concept1['tty']

                        rxcui2 = concept2['rxcui']
                        name2 = concept2['name']
                        tty2 = concept2['tty']

                        # Extract source concept information
                        source_concept = interaction_pair['interactionConcept'][0]['sourceConceptItem']
                        source_id = source_concept['id']
                        source_name = source_concept['name']
                        #source_url = source_concept['urlLocation']

                        # Now you can process and print the extracted information as needed
                        print(f"Source Name: {source_name}")
                        print(f"Comment: {comment}")
                        #print(f"Severity: {severity}")
                        print(f"RxCUI 1: {rxcui1}, Name 1: {name1}, TTY 1: {tty1}")
                        print(f"RxCUI 2: {rxcui2}, Name 2: {name2}, TTY 2: {tty2}")
                        print(f"Source ID: {source_id}")
                        print(f"Source Name: {source_name}")
                        #print(f"Source URL: {source_url}")

        else:
            print("No drug interactions found.")"""

    else:
        print(f"Error: Request failed with status code {response.status_code}")
        

except requests.exceptions.RequestException as e:
    print(f"Error making the request: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")