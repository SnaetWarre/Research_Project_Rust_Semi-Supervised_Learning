# Foreword

This thesis is the closing piece of my bachelor's programme in Multimedia and Creative Technologies at Howest University of Applied Sciences in Kortrijk. It was written for the Research Project module during my final semester and tackles a question I kept returning to throughout my studies: whether machine learning can realistically run on a phone, fully offline, in a way that lets the system keep learning over time without depending on a server or a connection.

The starting point was a certain frustration with how ML systems are usually presented. Almost every tutorial, benchmark and paper assumes that there is a server, a stable connection and essentially unlimited compute. That works well enough as a research baseline, but it does not match what many real deployments actually look like, especially in agriculture, infrastructure monitoring or any setting where connectivity cannot be taken for granted. I became genuinely curious about what it would take to close that gap.

Plant disease detection turned out to be a good fit for this kind of question. The PlantVillage dataset is publicly available and well documented, the economic stakes behind early disease detection are real and measurable, and the distance between existing solutions (cloud-dependent or lab-based) and what farmers in rural regions actually have access to is concrete. Together, those three factors made it a natural topic for research on offline, edge-deployable machine learning.

Choosing Rust for the whole implementation was both deliberate and, to some extent, experimental. I had been exploring the language on my own for about a year before starting this project, and I wanted to see whether the ML ecosystem around it had matured enough to act as a genuine alternative to Python for this kind of work. The answer, as this thesis documents in detail, is that it largely has, with a few specific limitations that are worth understanding before the choice is made.

I would like to thank [TODO: internal promoter name] for the consistent feedback throughout the research and for the practical guidance on scoping the work. I also want to thank [TODO: external promoter name] for reading the drafts critically and for pushing back on certain conclusions in a way that made them stronger. [TODO: thank external interview contacts by name here, once interviews are completed]

Finally, a word of thanks to my fellow students in the MCT programme for the continuous exchange of ideas and the general encouragement over the past year, and to my family for their patience during the months when free time was in short supply.

Warre Snaet
Kortrijk, April 2026
