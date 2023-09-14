# OSKAR Telescope Model

## SKA1-LOW

Ref : SKA1-LOW_SKO-0000422_Rev3_38m_SKALA4_spot_frequencies.tm from ska-simulation



## File Types

| file name    | Meaning                                      | Required |
|--------------|----------------------------------------------|----------|
| position.txt | Centre reference positino of telescope array | yes      |
| layout.txt   |The layout (in horizontal East-North-Up coordinates) of stations or elements within stations.| yes      |


### layout.txt

The file contents:




## TBA

layout_ecef.txt
The layout of stations in Earth-centred-Earth-fixed coordinates.
Can be used instead of "layout.txt" or "layout_wgs84.txt" at top-level only, if required.
See Telescope Level Earth-centred Coordinates
Required: No, unless layout.txt and layout_wgs84.txt are omitted.
Allowed locations: Telescope model root directory.

layout_wgs84.txt
The layout of stations in WGS84 (longitude, latitude, altitude) coordinates.
Can be used instead of "layout.txt" or "layout_ecef.txt" at top-level only, if required.
See Telescope Level WGS84 Coordinates
Required: No, unless layout.txt and layout_ecef.txt are omitted.
Allowed locations: Telescope model root directory.

element_types.txt
Type index of each element in the station.
See Element Types
Required: No.
Allowed locations: Station directory.

gain_phase.txt
Per-element gain and phase offsets and errors.
See Element Gain & Phase Error Files
Required: No.
Allowed locations: Station directory.

cable_length_error.txt
Per-element cable length errors.
See Element Cable Length Error Files
Required: No.
Allowed locations: Station directory.


apodisation.txt | apodization.txt
Per-element complex apodisation weight.
See Element Apodisation Files
Required: No.
Allowed locations: Station directory.

feed_angle.txt | feed_angle_x.txt | feed_angle_y.txt
Per-element and per-polarisation feed angles.
See Element Feed Angle Files
Required: No.
Allowed locations: Station directory.

permitted_beams.txt
Permitted station beam directions relative to mounting platform.
See Permitted Beam Directions
Required: No.
Allowed locations: Station directory.

element_pattern_fit_∗.bin
Fitted element X-or Y-dipole responses for the station, as a function of frequency.
See Numerical Element Patterns
Required: No.
Allowed locations: Any. (Inherited.)


element_pattern_spherical_wave_∗.txt
Fitted spherical wave element coefficient data, as a function of frequency.
See Spherical Wave Element Patterns
Required: No.
Allowed locations: Any. (Inherited.

noise_frequencies.txt
Frequencies for which noise values are defined.
See System Noise Configuration Files
Required: No, unless another noise file is present.
Allowed locations: Telescope model root directory. (Inherited.)

rms.txt
Flux density RMS noise values, in Jy, as a function of frequency.
See System Noise Configuration Files
Required: No.
Allowed locations: Telescope model root directory, or top-level station directory. (Inherited.)



