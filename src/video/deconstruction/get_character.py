"""
Character identification and subtitle enhancement script.
This script analyzes subtitles to identify characters and replace speaker labels with actual character names.
"""

import os
import re
import json
import litellm
from typing import Dict, List, Tuple, Optional

from src.prompt import CHARACTER_IDENTIFICATION_PROMPT


def parse_srt(srt_path: str) -> List[Dict]:
    """
    Parse an SRT subtitle file into a list of subtitle entries.

    Args:
        srt_path: Path to the SRT file

    Returns:
        List of dictionaries containing subtitle information
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\n+', content.strip())

    subtitles = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
            timestamp = lines[1]
            text = '\n'.join(lines[2:])

            # Extract speaker label if present
            speaker_match = re.match(r'\[([^\]]+)\]\s*(.*)', text, re.DOTALL)
            if speaker_match:
                speaker = speaker_match.group(1)
                dialogue = speaker_match.group(2).strip()
            else:
                speaker = "UNKNOWN"
                dialogue = text.strip()

            subtitles.append({
                'index': index,
                'timestamp': timestamp,
                'speaker': speaker,
                'dialogue': dialogue,
                'raw_text': text
            })
        except (ValueError, IndexError):
            continue

    return subtitles


def get_speaker_dialogues(subtitles: List[Dict]) -> Dict[str, List[str]]:
    """
    Group dialogues by speaker.

    Args:
        subtitles: List of parsed subtitle entries

    Returns:
        Dictionary mapping speaker labels to their dialogues
    """
    speaker_dialogues = {}
    for sub in subtitles:
        speaker = sub['speaker']
        if speaker not in speaker_dialogues:
            speaker_dialogues[speaker] = []
        if sub['dialogue']:  # Only add non-empty dialogues
            speaker_dialogues[speaker].append(sub['dialogue'])
    return speaker_dialogues


def estimate_tokens(text: str) -> int:
    """
    Estimate token count. Rough heuristic: ~3-4 chars per token for English.
    """
    return len(text) // 3


def format_dialogues_for_analysis(
    speaker_dialogues: Dict[str, List[str]],
    max_samples: int = None,  # None means all samples
    max_total_tokens: int = 50000  # Leave room for prompt template and response
) -> str:
    """
    Format speaker dialogues for LLM analysis.

    Args:
        speaker_dialogues: Dictionary of speaker to dialogues
        max_samples: Maximum samples per speaker (None = all)
        max_total_tokens: Maximum estimated tokens

    Returns:
        Formatted string for LLM prompt
    """
    formatted = []
    for speaker, dialogues in sorted(speaker_dialogues.items()):
        if speaker == "UNKNOWN":
            continue
        # Take all or limited samples
        samples = dialogues if max_samples is None else dialogues[:max_samples]
        sample_text = ' | '.join(samples)
        formatted.append(f"[{speaker}]: {sample_text}")

    result = '\n'.join(formatted)

    # Estimate and warn about token count
    estimated_tokens = estimate_tokens(result)
    print(f"  📝 [Character] Dialogue text length: {len(result)} chars, ~{estimated_tokens} tokens")

    if estimated_tokens > max_total_tokens:
        print(f"  ⚠️  [Character] WARNING: Estimated tokens ({estimated_tokens}) exceeds limit ({max_total_tokens})")
        print(f"  💡 [Character] Consider using max_samples parameter to limit dialogue samples")

    return result


def format_full_subtitles(subtitles: List[Dict]) -> str:
    """
    Format full subtitles for LLM analysis (preserves conversation flow).

    Args:
        subtitles: List of parsed subtitle entries

    Returns:
        Formatted string with full subtitle content
    """
    lines = []
    for sub in subtitles:
        speaker = sub['speaker']
        dialogue = sub['dialogue']
        if dialogue.strip():  # Skip empty dialogues
            lines.append(f"[{speaker}] {dialogue}")

    result = '\n'.join(lines)
    estimated_tokens = estimate_tokens(result)
    print(f"  📝 [Character] Full subtitle length: {len(result)} chars, ~{estimated_tokens} tokens")

    return result


def query_vllm_for_characters(
    dialogues_text: str,
    movie_name: str = "",
    model: str = None,
    api_base: str = None,
    api_key: str = "EMPTY",
    max_tokens: int = None,
) -> Dict:
    """
    Query LLM to identify characters from their dialogues.

    Args:
        dialogues_text: Formatted dialogue text
        movie_name: Optional movie name for context
        model: LLM model name
        api_base: API base URL
        api_key: API key
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary mapping speaker labels to character information
    """
    movie_context = f"This is from the movie '{movie_name}'." if movie_name else ""
    prompt = CHARACTER_IDENTIFICATION_PROMPT.format(
        movie_context=movie_context,
        dialogues_text=dialogues_text,
    )

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
            api_base=api_base,
            api_key=api_key,
        )
        content = response.choices[0].message.content

        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                print(f"❌ [Character] Error parsing JSON response: {e}")
                return {}
        else:
            print(f"⚠️  [Character] Could not parse JSON from response: {content[:500]}")
            return {}

    except Exception as e:
        print(f"❌ [Character] Error querying LLM: {e}")
        return {}


def refine_character_mapping(
    character_info: Dict,
    speaker_dialogues: Dict[str, List[str]]
) -> Dict[str, str]:
    """
    Refine character mapping and handle edge cases.

    Args:
        character_info: Raw character identification results
        speaker_dialogues: Original speaker to dialogues mapping

    Returns:
        Clean mapping of speaker labels to character names
    """
    mapping = {}

    for speaker, info in character_info.items():
        if isinstance(info, dict):
            name = info.get('name', 'Unknown')
            confidence = info.get('confidence', 'low')

            # If confidence is low and name is Unknown, keep original label
            if name == 'Unknown' and confidence == 'low':
                mapping[speaker] = speaker
            else:
                mapping[speaker] = name
        else:
            mapping[speaker] = str(info) if info else speaker

    # Add any speakers not in the response
    for speaker in speaker_dialogues.keys():
        if speaker not in mapping:
            mapping[speaker] = speaker

    return mapping


def create_new_subtitles(
    subtitles: List[Dict],
    speaker_mapping: Dict[str, str]
) -> List[Dict]:
    """
    Create new subtitle entries with character names instead of speaker labels.

    Args:
        subtitles: Original subtitle entries
        speaker_mapping: Mapping from speaker labels to character names

    Returns:
        New subtitle entries with character names
    """
    new_subtitles = []

    for sub in subtitles:
        new_sub = sub.copy()
        original_speaker = sub['speaker']
        character_name = speaker_mapping.get(original_speaker, original_speaker)

        # Update the raw text with the character name
        new_sub['speaker'] = character_name
        new_sub['raw_text'] = f"[{character_name}] {sub['dialogue']}"

        new_subtitles.append(new_sub)

    return new_subtitles


def write_srt(subtitles: List[Dict], output_path: str):
    """
    Write subtitles to an SRT file.

    Args:
        subtitles: List of subtitle entries
        output_path: Path to output SRT file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sub in subtitles:
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['timestamp']}\n")
            f.write(f"{sub['raw_text']}\n")
            f.write("\n")

    print(f"💾 [Character] Saved new subtitles to: {output_path}")


def write_character_info(character_info: Dict, output_path: str):
    """
    Write character information to a JSON file.

    Args:
        character_info: Character identification results
        output_path: Path to output JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(character_info, f, ensure_ascii=False, indent=2)

    print(f"💾 [Character] Saved character info to: {output_path}")


def analyze_subtitles(
    srt_path: str,
    movie_name: str = "",
    output_dir: Optional[str] = None,
    use_full_subtitles: bool = True,
    model: str = None,
    api_base: str = None,
    api_key: str = "EMPTY",
    max_tokens: int = None,
) -> Tuple[Dict[str, str], Dict]:
    """
    Main function to analyze subtitles and identify characters.

    Args:
        srt_path: Path to the SRT subtitle file
        movie_name: Name of the movie (optional, for better context)
        output_dir: Directory to save output files (optional)
        use_full_subtitles: If True, send full subtitles (better accuracy);
                           If False, send grouped by speaker (less tokens)
        model: LLM model name
        api_base: API base URL
        api_key: API key
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (speaker_mapping, character_info)
    """
    print(f"🔍 [Character] Analyzing subtitles: {srt_path}")

    # Parse subtitles
    subtitles = parse_srt(srt_path)
    print(f"✅ [Character] Parsed {len(subtitles)} subtitle entries")

    # Group dialogues by speaker
    speaker_dialogues = get_speaker_dialogues(subtitles)
    print(f"👥 [Character] Found {len(speaker_dialogues)} unique speakers:")
    for speaker, dialogues in sorted(speaker_dialogues.items()):
        print(f"  {speaker}: {len(dialogues)} dialogues")

    # Format dialogues for analysis
    if use_full_subtitles:
        print("\n✨ [Character] Using FULL subtitles for analysis (better accuracy)...")
        dialogues_text = format_full_subtitles(subtitles)
    else:
        print("\n⚡ [Character] Using grouped dialogues for analysis (less tokens)...")
        dialogues_text = format_dialogues_for_analysis(speaker_dialogues)

    # Query LLM for character identification
    print("\n🤖 [Character] Querying LLM for character identification...")
    character_info = query_vllm_for_characters(
        dialogues_text,
        movie_name,
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
    )

    if character_info:
        print("\n🎭 [Character] Identified characters:")
        for speaker, info in sorted(character_info.items()):
            if isinstance(info, dict):
                name = info.get('name', 'Unknown')
                confidence = info.get('confidence', 'unknown')
                role = info.get('role', 'unknown')
                evidence = info.get('evidence', '')
                print(f"  {speaker} -> {name} (confidence: {confidence}, role: {role})")
                print(f"    Evidence: {evidence[:100]}...")
            else:
                print(f"  {speaker} -> {info}")

    # Create clean mapping
    speaker_mapping = refine_character_mapping(character_info, speaker_dialogues)

    # Save outputs if output_dir is specified
    if output_dir is None:
        output_dir = os.path.dirname(srt_path)

    # Create new subtitles with character names
    new_subtitles = create_new_subtitles(subtitles, speaker_mapping)

    # Write new subtitle file
    base_name = os.path.splitext(os.path.basename(srt_path))[0]
    new_srt_path = os.path.join(output_dir, f"{base_name}_with_characters.srt")
    write_srt(new_subtitles, new_srt_path)

    # Write character info JSON
    char_info_path = os.path.join(output_dir, "character_info.json")
    write_character_info(character_info, char_info_path)

    # Write speaker mapping JSON
    mapping_path = os.path.join(output_dir, "speaker_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_mapping, f, ensure_ascii=False, indent=2)
    print(f"💾 [Character] Saved speaker mapping to: {mapping_path}")

    return speaker_mapping, character_info

