# Generation Samples

Checkpoint: downloaded batch-128 scaled-LR checkpoint

Prompt: `Once upon a time, there was a little girl named Lily`

Max new tokens: `256`

## Conclusion

The trained model clearly learned TinyStories style: simple vocabulary, short
sentences, a child-centered plot, a small conflict, a resolution, and clean
`<|endoftext|>` stopping. The strongest decoding setting here is
`temperature=0.8` with `top_p=0.9`, which gives the best balance between fluency
and variety. Greedy decoding is also coherent, but it is more generic and
template-like.

Increasing sampling randomness improves variety but weakens consistency.
`temperature=1.0` with `top_p=0.9` remains readable, but introduces small causal
oddities such as the mosquito and ball sequence. `temperature=1.1` with
`top_p=0.95` is the weakest sample because it drifts semantically across the
magic pencil, ball, school, and microscope. Overall, the batch-128 scaled-LR
checkpoint is good enough to produce coherent short stories, but the model still
leans on generic story patterns and becomes less reliable as decoding becomes
more adventurous.

## greedy

Temperature: `0.0`; top-p: `none`; seed: `11`

```text
Once upon a time, there was a little girl named Lily. She loved to play with her toys and have fun. One day, she found a big box in her room. She was very excited to see what was inside.
Lily opened the box and found a big, soft teddy bear. She was so happy! She hugged the teddy bear and said, "Thank you, teddy bear!" But then, something unexpected happened. The teddy bear started to talk!
The teddy bear said, "I am a magic teddy bear. I can make your wishes come true!" Lily was very surprised. She wished for a big ice cream cone. The teddy bear made her wish come true. Lily and the teddy bear became best friends and had many fun adventures together.
<|endoftext|>
```

## t0.8_p0.9

Temperature: `0.8`; top-p: `0.9`; seed: `12`

```text
Once upon a time, there was a little girl named Lily. She had a toy that could shoot soft balls. One day, Lily went outside to play with her ball. But, the ball was broken. Lily was sad.
Lily's mom saw her sad face and asked, "Why are you sad, Lily?" Lily showed her mom the broken ball. Her mom said, "Don't worry, I will repair it for you." Lily was happy to hear that.
Mom took Lily to the toy store. They got the ball and went back home. Lily played with her ball all day. She was careful not to shoot it too hard. And they all lived happily ever after.
<|endoftext|>
```

## t1.0_p0.9

Temperature: `1.0`; top-p: `0.9`; seed: `13`

```text
Once upon a time, there was a little girl named Lily. She lived in a small house with her mom, dad, and her dog, Max. Lily loved to play outside and have fun with Max. But Max was very lazy and always wanted to sleep.
One day, Lily went to the garden to play. She found a big, round ball and decided to play with it. Max was happy and wagged his tail. But then, Lily saw a mosquito on the ball. She had an idea. Lily yelled at the mosquito, "Go away, mosquito!"
The mosquito flew to Lily and tried to bite her. Max saw the mosquito and started to chase it away. The mosquito was fast and caught the ball. But then, Max started to play with the ball too hard and too high. He picked it up and brought it back to Lily.
"Thank you, Max!" said Lily. "You are a good friend." Max was happy to help and they played together all day. From that day on, Lily and Max were the best of friends.
<|endoftext|>
```

## t1.1_p0.95

Temperature: `1.1`; top-p: `0.95`; seed: `14`

```text
Once upon a time, there was a little girl named Lily. She had a magic pencil that could do amazing things. Lily liked to act and make wishes with her mom.
One day, Lily and her mom were using the magic pencil. Lily wanted to choose a big, red ball to play with. Her mom said she could choose a small ball if she helped her win a big, red ball. Lily was so happy!
Lily started to think about the secret one day. She imagined a big ball she knew at school. Lily told her friends about the happy times they found in her microscope. They knew that Lily was special, even with her incredible big, red ball.
<|endoftext|>
```
