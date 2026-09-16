"""Priprema usklađenih izlaza ručne korekcije bez ponovnog prepoznavanja zvuka."""
import json
from copy import deepcopy
from pathlib import Path

from .medical_summary import create_medical_summary
from .clinical_record import create_clinical_record


def prepare_revision_outputs(transcript_path: Path, new_text: str, revision: int) -> dict[Path, str]:
    """Validiraj redak-po-iskaz izmjenu i izradi sve izvedene sadržaje prije zapisivanja."""
    structured_path = transcript_path.with_name(transcript_path.stem + '_strukturirano.json')
    if not structured_path.exists():
        # Stariji samostalni TXT nema izvornu vremensku referencu koju bi se smjelo izmišljati.
        derived = [transcript_path.with_name(transcript_path.stem + suffix)
                   for suffix in ('_medicinski_sazetak.json', '_medicinski_sazetak.txt',
                                  '_lijecnicki_zapis.json', '_lijecnicki_zapis.txt')]
        if any(path.exists() for path in derived):
            raise ValueError('Nedostaje strukturirani transkript potreban za osvježavanje sažetka.')
        return {}
    payload = json.loads(structured_path.read_text(encoding='utf-8'))
    utterances = deepcopy(payload['utterances'])
    lines = new_text.splitlines()
    if len(lines) != len(utterances):
        raise ValueError('Sačuvajte jedan redak po izvornom iskazu. Dodavanje ili brisanje redaka zahtijeva novo vremensko poravnanje.')
    aliases = {'DOKTOR': 'DOKTOR', 'LIJEČNIK': 'DOKTOR', 'LIJECNIK': 'DOKTOR',
               'PACIJENT': 'PACIJENT', 'NEPOZNATO': 'NEPOZNATO'}
    for line, item in zip(lines, utterances):
        label, separator, text = line.partition(':')
        if not separator or not text.strip():
            raise ValueError('Svaki redak mora imati oblik ULOGA: tekst i neprazan iskaz.')
        label = label.strip()
        role = aliases.get(label.upper())
        if role is None and label not in {item.get('role'), item.get('speaker')}:
            raise ValueError(f'Nepoznata oznaka govornika: {label}')
        original_text = item['text']
        item.setdefault('raw_text', original_text)
        item['text'] = text.strip()
        item['role'] = role or label
        item['manual_revision'] = revision
        if item['text'] != original_text:
            # Stara vremena riječi vrijede za izvorni ASR, ne za novoupisane riječi.
            if 'words' in item:
                item.setdefault('original_words', deepcopy(item['words']))
                item.pop('words')
            item['word_alignment_status'] = 'requires_realignment_after_manual_edit'
    payload['utterances'] = utterances
    payload['manual_revision'] = revision
    summary = create_medical_summary(utterances, source={
        'audio': payload.get('audio'), 'structured_transcript': str(structured_path),
        'model': payload.get('model'), 'language': payload.get('language', 'hr'),
        'manual_revision': revision,
    })
    summary['generator']['refresh_reason'] = 'manual_transcript_revision'
    record = create_clinical_record(summary)
    def encoded(value):
        """Sačuvaj hrvatska slova u čitljivom JSON prikazu."""
        return json.dumps(value, ensure_ascii=False, indent=2)
    return {
        structured_path: encoded(payload),
        transcript_path.with_name(transcript_path.stem + '_medicinski_sazetak.json'): encoded(summary),
        transcript_path.with_name(transcript_path.stem + '_medicinski_sazetak.txt'): summary['summary_text'],
        transcript_path.with_name(transcript_path.stem + '_lijecnicki_zapis.json'): encoded(record),
        transcript_path.with_name(transcript_path.stem + '_lijecnicki_zapis.txt'): record['record_text'],
    }
