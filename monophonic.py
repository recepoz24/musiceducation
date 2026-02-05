import muspy
# Read the MIDI file
music = muspy.read_midi("rondo_alla_turca.mid")

# Select the first track (typically contains the main melody)
track = music.tracks[0]

# Sort notes by their start time
notes = sorted(track.notes, key=lambda n: n.time)

# Create a monophonic sequence: only one note at each time point (e.g., the first encountered)
monophonic_notes = []
current_time = -1

for note in notes:
    if note.time != current_time:
        monophonic_notes.append(note)
        current_time = note.time  # avoid simultaneous notes (monophonic)

# Extract only the pitch values from the monophonic sequence
monophonic_pitches = [note.pitch for note in monophonic_notes]

# Output the result
print("Monophonic sequence (pitch):", monophonic_pitches)
