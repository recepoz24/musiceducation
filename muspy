import muspy
import collections

# Load the MIDI file
music = muspy.read_midi("rondo_alla_turca.mid")

# Melody: Extract the pitch values of the first 10 notes (assuming a monophonic main melody)
melody_track = music[0]  # the main melody is typically in the first track
melody_pitches = [note.pitch for note in melody_track.notes[:10]]
print("First 10 Melody Pitch Values:", melody_pitches)

# Chords: Identify the first 5 groups of notes played simultaneously
chords = []
sorted_notes = sorted(melody_track.notes, key=lambda n: n.time)
current_chord = [sorted_notes[0]]
for note in sorted_notes[1:]:
    if abs(note.time - current_chord[0].time) < 10:  
# very small time difference = simultaneous notes
        current_chord.append(note)
    else:
        if len(current_chord) > 1:
            chords.append([n.pitch for n in current_chord])
        current_chord = [note]
    if len(chords) >= 5:
        break

print("First 5 Chords:", chords)

# Time signatures: Extract the first 5 time signatures from the score
if music.time_signatures:
    first_5_time_signatures = music.time_signatures[:5]
    time_signs = []
    for ts in first_5_time_signatures:
        time_signs.append((ts.numerator, ts.denominator))
else:
    time_signs = []  # If there are no time signature events

print("First 5 Time Signatures:", time_signs)
