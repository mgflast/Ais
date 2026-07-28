# Linking Ais and Pom

[Pom](https://github.com/mgflast/Pom) is a companion tool for working with segmentations at scale. It reuses the Ais rendering engine, runs on an HPC cluster, and turns a whole dataset into a searchable, browsable database of tomograms and their segmentations. Ais synchronises with it, so you can browse a dataset in Pom and open any tomogram or model straight into Ais.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/pom_database.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Left: inspecting the output of a preliminary model on several tomograms in Ais. Right: the Pom app running in the browser, connected to Ais, used to explore the dataset and open tomograms of interest.</p>

## Synchronising Ais and Pom

Turn on **Settings → Pom → Synchronize Ais & Pom**, then set a **command directory** (**Settings → Pom → Set command directory**). The sync runs entirely through that shared directory: when you click *open in Ais* on a tomogram or a model in Pom, Pom writes a command file into the directory, and Ais — which watches it — opens that tomogram or model. Ais needs access to the command directory wherever it runs; Pom can run anywhere, as long as it writes somewhere Ais can read.

Because Ais is the interactive half, it is usually best to run it locally, so there is no input latency. A typical setup is Pom on a cluster and Ais on your own machine: have Pom write its commands to, say, `~/Ais` on the cluster, map that location as a network drive on your local machine, and set the same mapped folder as Ais's command directory. Pom then writes commands from the cluster, and your local Ais picks them up.

## More about Pom

Pom also shares the `<tomogram>__<model>.mrc` filename convention with Ais, and a Pom subset file can be handed to [`ais pick`](../cli/reference.md#ais-pick) via `--subset` to restrict picking to a chosen set of tomograms. See the [Pom documentation](https://mgflast.github.io/Pom/) for more <!-- placeholder: Pom docs not yet published -->.
