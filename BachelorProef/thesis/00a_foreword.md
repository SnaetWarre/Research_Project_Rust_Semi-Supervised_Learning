# Foreword

This thesis is the concluding work of my bachelor's programme in Multimedia and Creative Technologies at Howest University of Applied Sciences in Kortrijk. It was written as part of the Research Project module in my final semester and investigates a question that I kept coming back to throughout my studies: whether machine learning can realistically run on a phone, fully offline, in a way that the system itself can continue to learn over time without needing a server or a connection.

The starting point was a kind of frustration with how ML systems are usually presented. Almost every tutorial, every benchmark, and every paper assumes a server, a stable connection, and essentially unlimited compute. That works fine as a research baseline but it does not match what a lot of real deployments look like, especially in agriculture, infrastructure monitoring, or any setting where connectivity is not guaranteed. I became genuinely curious about what it would actually take to close that gap.

Plant disease detection turned out to be a well-suited domain for exactly this kind of research. The PlantVillage dataset is publicly available and well-documented, the economic stakes behind early disease detection are real and significant, and the gap between current solutions (cloud-dependent or lab-based) and what farmers in rural areas actually have access to is concrete and verifiable. Those factors made it a natural fit for a research question about offline, edge-deployable machine learning.

The decision to implement everything in Rust was both deliberate and somewhat experimental. I had been exploring the language on my own for about a year before starting this project, and I wanted to find out whether the ecosystem around ML in Rust had matured to the point where it could serve as a genuine alternative to Python for this kind of work. The answer, as this thesis documents in detail, is that it largely can, with a few specific limitations that are worth understanding before making the choice.

I want to thank [TODO: internal promoter name] for the consistent feedback throughout the research process and for the practical guidance on how to scope the work. I also want to thank [TODO: external promoter name] for the critical reading of the drafts and for pushing back on certain conclusions in a way that made them stronger. [TODO: thank external interview contacts by name here, once interviews are completed]

Thanks also to my fellow students in the MCT programme for the ongoing exchange of ideas and the general encouragement over the past year, and to my family for their patience during the months when free time was hard to come by.

Warre Snaet
Kortrijk, April 2026
