# Foreword

This thesis is the closing piece of my bachelor's programme in Multimedia and Creative Technologies at Howest University of Applied Sciences in Kortrijk. It was written for the Research Project module during my final semester. The work started from a question I kept coming back to during my studies: can machine learning realistically run on a phone, fully offline, while still allowing the system to improve over time without depending on a server or a connection?

The starting point was a certain frustration with how ML systems are usually presented. Almost every tutorial, benchmark and paper assumes that there is a server, a stable connection and almost unlimited compute. That works as a research baseline, but it does not match many real deployments, especially in agriculture, infrastructure monitoring or other settings where connectivity cannot be taken for granted. I became curious about what it would take to close that gap.

Plant disease detection turned out to be a good fit for this kind of question. The PlantVillage dataset is publicly available and well documented, the economic stakes behind early disease detection are real and measurable, and the distance between existing solutions (cloud-dependent or lab-based) and what farmers in rural regions have access to is concrete. Together, those three factors made it a natural topic for research on offline, edge-deployable machine learning.

Choosing Rust for the whole implementation was deliberate, but also a bit experimental. I had been exploring the language on my own for about a year before starting this project, and I wanted to see whether the ML ecosystem around it had matured enough to be a real alternative to Python for this kind of work. This thesis documents the answer in detail: it mostly has, though a few specific limitations are worth understanding before you commit to that choice.

I would like to thank Gilles Depypere for the consistent feedback throughout the research and for the practical guidance on scoping the work. I also want to thank Sandro Queirós for reading the drafts critically and for pushing back on certain conclusions in a way that made them stronger. I also want to thank Helena Torres and Pedro Morais from 2AI-IPCA for reviewing the technical approach and sharing their expertise on image classification, deployment and data augmentation.

Finally, I want to thank my fellow students in the MCT programme for the many exchanges of ideas and for the encouragement during the past year. I also want to thank my family for their patience during the months when free time was limited.

Warre Snaet
Kortrijk, April 2026
