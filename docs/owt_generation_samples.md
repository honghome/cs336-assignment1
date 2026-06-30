# OpenWebText Generation Samples

Checkpoint: downloaded OWT checkpoint from run `orange_camel_9jgt9nv15m`

Prompt: `The United States government`

Max new tokens: `192`

## Conclusion

The OWT checkpoint learned a recognizable web/news style: paragraphs, political
entities, quotations, topic shifts, and public-affairs vocabulary. It is much
less clean than the TinyStories checkpoint because the data distribution is much
broader and the model is still small. Greedy decoding is fluent but repetitive,
especially around military/government phrasing. `temperature=0.8` with
`top_p=0.9` gives the best balance here: it avoids the worst repetition and
produces article-like prose, though the facts are not reliable. Higher
temperature increases variety but also produces citation-like clutter,
inconsistent entities, and semantic drift.

## greedy

Temperature: `0.0`; top-p: `none`; seed: `11`

```text
The United States government has been forced to take action against the U.S. government in the face of the U.S. government’s decision to withdraw its troops from the U.S. military.

The U.S. government has been in the process of a new military operation in Afghanistan, and the U.S. has been in the process of launching a new military operation in Afghanistan.

The U.S. military has been in the process of launching a new military operation in Afghanistan, and the U.S. has been in the process of launching a new military operation in Afghanistan.

The U.S. military has been in the process of launching a new military operation in Afghanistan, and the U.S. has been in the process of launching a new military operation in Afghanistan.

The U.S. military has
```

## t0.8_p0.9

Temperature: `0.8`; top-p: `0.9`; seed: `12`

```text
The United States government has not been able to define who has come into the world in recent years. As a result, the United States can be nearly ready for a public relations, to have a leader in an unprecedented international push to protect its border.

The fight against terrorism is part of the global jihad. Even the Islamic State, one of the most fundamental powers, has been a threat to the Middle East. The United States has deployed a powerful arms security in Iraq, Lebanon, Afghanistan, Iran, and the United States for the first time since 2003. The United States has been an ally of Assad’s Islamic State, but it is in the fight to create a safe haven for a self-reliance.

Russia is also the only American opposition to the Middle East. It has been clear that the United States is not ready to confront ISIS. But the US has always been able to carry its own weapons. It has been battling for more than a decade
```

## t1.0_p0.9

Temperature: `1.0`; top-p: `0.9`; seed: `13`

```text
The United States government with the best measure of sight, thanks to the bipartisan selection of bipartisan bodies in its own right. That’s what’s in line with presidents and presidents and military officials.

Obama, who served as President-elect after meeting his former senator, has had multiple US troops run by Congress. He has received the Presidential Medal for Freedom of Religion in support of the American Conservative Party, a diverse group with numerous positions in Washington. His declaration of unity is a long way from the back of the war at the expense of the President and his army. His movement by the power will be practiced by the Democratic Party.

As the Navy continues to release the UN Security Council Resolution, Members of Congress have the ability to scale everything. President-elect Obama said he will vote to enhance freedom of assembly to lead in support of the United States.

"He will never give America great authority on this territory, this country, and I think he
```

## t1.1_p0.95

Temperature: `1.1`; top-p: `0.95`; seed: `14`

```text
The United States government, the black tractor, the Hispanic horizontal rate of the French gillor, religious groups and the people.[70]

The Claims of Thomas' phone,[71] other libraries,[72] or the documents that make up the data base and address the level of some of the weight of the CVSON rates. When Frederick was absent multiple media market activity he fed glasses to. "Here in New York, we used computers just mine and Microsoft for book detailing the U.S. flag. So as the above guys have been filled with a vast array of devices on the Internet, Matt, and the smart guy cares," Branner testified. Under surveillance framework, Wildstein alone became the greatest file-exposed American machine in the United States, such as Berkeley- has the only optical video done by the automobile pioneers... The same can be summed up by some Johnson Parties and Mexico and Americans from state would be clearly a little
```