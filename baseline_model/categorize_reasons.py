import pandas as pd
import re
import json

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Dataset/DSU-Dataset.csv')
reasons = df['REASON_VISIT_NAME'].dropna().unique()

print(f"Total unique reasons: {len(reasons)}")

# Define categorization rules based on keywords and patterns
categories = {
    'Injury/Trauma': [
        r'\b(FALL|FRACTURE|BREAK|DISLOCATION|SPRAIN|STRAIN|CONTUSION|BRUISE|LACERATION|CUT|PUNCTURE|WOUND|SCRATCH|ABRASION|TRAUMA|INJURY|AMPUTATION)\b',
        r'\b(HEAD INJURY|CLOSED HEAD|CONCUSSION|TBI)\b',
        r'\b(FOOT INJURY|ANKLE INJURY|KNEE INJURY|HAND INJURY|ARM INJURY|LEG INJURY|NECK INJURY|BACK INJURY)\b',
        r'\b(BURN|SCALD|FROSTBITE|COLD EXPOSURE|HEAT EXPOSURE)\b',
        r'\b(ASSALT|ASSAULT|VICTIM)\b',
        r'\b(MVC|MOTOR VEHICLE|CAR ACCIDENT|MVA)\b',
        r'\b(DOG BITE|ANIMAL BITE|BITE)\b',
    ],
    'Mental Health/Behavioral': [
        r'\b(ANXIETY|DEPRESSION|SUICIDAL|SUICIDE|MENTAL|PSYCH|PSYCHIATRIC|BEHAVIOR|BEHAVIOUR|PSYCHOSIS|BIPOLAR|PERSONALITY DISORDER)\b',
        r'\b(DRUG OVERDOSE|OVERDOSE|OD|POISONING|INTOXICATION|ALCOHOL|ETOH)\b',
        r'\b(SEXUAL ASSAULT|RAPE|EVALUATION OF SEXUAL)\b',
        r'\b(SHAKING|TREMOR|SEIZURE|CONVULSION)\b',
    ],
    'Pain': [
        r'\b(PAIN)\b',
        r'\b(HEADACHE|MIGRAINE)\b',
        r'\b(CHEST PAIN)\b',
        r'\b(ABDOMINAL PAIN|STOMACH PAIN)\b',
        r'\b(BACK PAIN|SPINE|SPINAL)\b',
        r'\b(JOINT PAIN|ARTHRITIS)\b',
    ],
    'Cardiovascular': [
        r'\b(CHEST PAIN|CARDIAC|HEART|ARRHYTHMIA|ATRIAL|VENTRICULAR|PALPITATION)\b',
        r'\b(HYPERTENSION|HYPOTENSION|BLOOD PRESSURE|BP)\b',
        r'\b(SYNCOPE|NEAR SYNCOPE|FAINT|FAINTING|DIZZY|DIZZINESS|VERTIGO)\b',
        r'\b(HEART ATTACK|MI|MYOCARDIAL|CARDIO)\b',
        r'\b(EDEMA|SWELLING)\b',
    ],
    'Respiratory': [
        r'\b(BREATHING|RESPIRATORY|ASTHMA|COPD|EMPHYSEMA|BRONCHITIS)\b',
        r'\b(SHORTNESS OF BREATH|SOB|DYSPNEA|WHEEZING)\b',
        r'\b(COUGH|CONGESTION|NASAL|RHINORRHEA)\b',
        r'\b(PNEUMONIA|LUNG|PULMONARY)\b',
        r'\b(FLU|INFLUENZA)\b',
    ],
    'Gastrointestinal': [
        r'\b(ABDOMINAL|STOMACH|GASTRO|GI|NAUSEA|VOMITING|VOMIT|DIARRHEA|DIARRHOEA|CONSTIPATION)\b',
        r'\b(RECTAL|RECTUM|BOWEL|HEMATEMESIS|HEMATOCHEZIA|MELENA|GI BLEED|GASTROINTESTINAL BLEED)\b',
        r'\b(APPENDICITIS|GALLBLADDER|GALLSTONE|CHOLECYSTITIS|PANCREATITIS)\b',
    ],
    'Infectious Disease/Fever': [
        r'\b(FEVER|PYREXIA|TEMP)\b',
        r'\b(FLU|INFLUENZA|VIRUS|VIRAL)\b',
        r'\b(INFECTION)\b',
        r'\b(CELLULITIS|ABSCESS|BACTERIAL)\b',
        r'\b(URINARY|UTI|CYSTITIS|PYELONEPHRITIS)\b',
        r'\b(EAR INFECTION|OTITIS|SINUS|SINUSITIS)\b',
        r'\b(SEPSIS|SEPTIC)\b',
    ],
    'Neurological': [
        r'\b(SEIZURE|CONVULSION|EPILEPSY)\b',
        r'\b(HEADACHE|MIGRAINE|NEURALGIA)\b',
        r'\b(WEAKNESS|PARALYSIS|PARESTHESIA|NUMBNESS)\b',
        r'\b(STROKE|CVA|TIA|TRANSIENT ISCHEMIC)\b',
        r'\b(DIZZINESS|VERTIGO|SYNCOPE)\b',
        r'\b(ALZHEIMER|DEMENTIA|MEMORY)\b',
    ],
    'Dermatological': [
        r'\b(DERM|SKIN|RASH|HIVES|URTICARIA|ECZEMA|DERMATITIS)\b',
        r'\b(BURN|BITE|WOUND|LACERATION|CUT)\b',
        r'\b(CELLULITIS|ABSCESS)\b',
    ],
    'Genitourinary/Reproductive': [
        r'\b(VAGINAL|UTERINE|MENSTRUAL|MENOPAUSE|POSTMENOPAUSAL|DYSFUNCTIONAL UTERINE)\b',  # Check first before GI bleeding
        r'\b(URINARY|UTI|CYSTITIS|DYSURIA|BURNING|FREQUENCY|URGENCY)\b',
        r'\b(PELVIC|GYN|GYNECOLOGIC|PERIOD|PREGNANCY|PREGNANT|OB|OBSTETRIC)\b',
        r'\b(KIDNEY|RENAL|NEPHRO|FLANK|NEPHROLITHIASIS|KIDNEY STONE)\b',
        r'\b(TESTICULAR|PENILE|PROSTATE)\b',
    ],
    'Endocrine/Metabolic': [
        r'\b(DIABETES|DIABETIC|HYPERGLYCEMIA|HYPOGLYCEMIA|INSULIN|GLUCOSE|BLOOD SUGAR)\b',
        r'\b(THYROID|HYPERTHYROID|HYPOTHYROID)\b',
    ],
    'Musculoskeletal': [
        r'\b(PAIN)\b',  # General pain
        r'\b(ARTHRITIS|JOINT|JOINTS)\b',
        r'\b(BACK|SPINE|SPINAL|NECK|CERVICAL|LUMBAR)\b',
        r'\b(FRACTURE|BREAK|DISLOCATION)\b',
    ],
    'Allergic/Immunological': [
        r'\b(ALLERGY|ALLERGIC|ANAPHYLAXIS|REACTION)\b',
        r'\b(HIVES|URTICARIA|RASH)\b',
    ],
    'Surgical/Procedural': [
        r'\b(SURGICAL|SURGERY|POST-OP|POSTOP|POST OPERATIVE|POSTOPERATIVE)\b',
        r'\b(FOLLOWUP|FOLLOW-UP|FOLLOW UP|FOLLOW-UP)\b',
        r'\b(CATHETER|TUBE|FEEDING|G TUBE|NG TUBE)\b',
        r'\b(CAST|REMOVAL|PLACEMENT|DRAIN)\b',
    ],
    'Other/Unspecified': [
        r'\b(OTHER|UNKNOWN|UNSPECIFIED|GENERAL|EVALUATION)\b',
        r'\b(WEAKNESS|FATIGUE|MALAISE|DIZZY)\b',  # Dizzy moved here if not cardiovascular
        r'\b(ULTRASOUND|CT|MRI|X-RAY|IMAGING)\b',
        r'\b(TRANSFER|DISCHARGE|ADMISSION)\b',
    ]
}

# Compile regex patterns
compiled_categories = {}
for category, patterns in categories.items():
    compiled_categories[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

# Function to categorize a reason
def categorize_reason(reason_str):
    if pd.isna(reason_str) or reason_str == 'nan':
        return 'Other/Unspecified'
    
    reason_str = str(reason_str).upper()
    
    # Check categories in priority order (more specific first)
    # Genitourinary must come before Gastrointestinal to catch vaginal bleeding correctly
    priority_order = [
        'Mental Health/Behavioral',
        'Injury/Trauma',
        'Genitourinary/Reproductive',  # Check before GI to catch vaginal/uterine bleeding
        'Cardiovascular',
        'Respiratory',
        'Neurological',
        'Gastrointestinal',
        'Infectious Disease/Fever',
        'Dermatological',
        'Pain',
        'Musculoskeletal',
        'Endocrine/Metabolic',
        'Allergic/Immunological',
        'Surgical/Procedural',
        'Other/Unspecified'
    ]
    
    for category in priority_order:
        if category in compiled_categories:
            for pattern in compiled_categories[category]:
                if pattern.search(reason_str):
                    return category
    
    # Default if no match
    return 'Other/Unspecified'

# Categorize all reasons
print("\nCategorizing reasons...")
reason_to_category = {}
for reason in reasons:
    if pd.notna(reason):
        reason_to_category[reason] = categorize_reason(reason)

# Create category mapping DataFrame
category_mapping = pd.DataFrame({
    'REASON_VISIT_NAME': list(reason_to_category.keys()),
    'Category': list(reason_to_category.values())
}).sort_values('Category')

# Show category distribution
print("\n" + "=" * 80)
print("CATEGORY DISTRIBUTION")
print("=" * 80)
category_counts = category_mapping['Category'].value_counts()
for cat, count in category_counts.items():
    print(f"{cat}: {count} reasons")

# Show some examples from each category
print("\n" + "=" * 80)
print("EXAMPLES BY CATEGORY")
print("=" * 80)
for category in category_counts.index[:10]:  # Show top 10 categories
    examples = category_mapping[category_mapping['Category'] == category]['REASON_VISIT_NAME'].head(5).tolist()
    print(f"\n{category}:")
    for ex in examples:
        print(f"  - {ex}")

# Save mapping to CSV
category_mapping.to_csv('reason_categories.csv', index=False)
print("\n[OK] Saved categorization mapping to 'reason_categories.csv'")

# Save as JSON for easy lookup
json_mapping = {str(k): v for k, v in reason_to_category.items()}
with open('reason_categories.json', 'w') as f:
    json.dump(json_mapping, f, indent=2)
print("[OK] Saved categorization mapping to 'reason_categories.json'")

# Apply categories to full dataset and analyze
print("\n" + "=" * 80)
print("APPLYING CATEGORIES TO DATASET")
print("=" * 80)
df['Reason_Category'] = df['REASON_VISIT_NAME'].map(reason_to_category)

# Show volume by category
category_volume = df.groupby('Reason_Category').agg({
    'ED Enc': 'sum',
    'ED Enc Admitted': 'sum'
}).reset_index()
category_volume['Admission Rate'] = (category_volume['ED Enc Admitted'] / category_volume['ED Enc'] * 100).round(2)
category_volume = category_volume.sort_values('ED Enc', ascending=False)

print("\nVolume by Category:")
print(category_volume.to_string(index=False))

# Save volume analysis
category_volume.to_csv('category_volume_analysis.csv', index=False)
print("\n[OK] Saved volume analysis to 'category_volume_analysis.csv'")

print("\n" + "=" * 80)
print("CATEGORIZATION COMPLETE")
print("=" * 80)
