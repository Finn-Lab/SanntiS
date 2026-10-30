/*
 * sanntis generic task dispatcher. Launched by sanntis itself, one top-level run per fan-out step:
 *
 *   nextflow run <pkg>/nf/tasks.nf -c <pkg>/nf/tasks.config [-c <user config> ...] \
 *       --manifest <run>/<step>/tasks.tsv --run <run> --step <step> --python <sys.executable>
 *
 * One process per manifest row (unit_id, label). Resources come from the row's label (stable interface:
 * cpu_small, train, gpu, antismash); defaults in tasks.config, overridden by the user's config (withLabel).
 * Python labels run `<python> -m sanntis task <step> --run <run> --unit <unit_id>` in the host env; the
 * antismash label runs the unit's script <run>/<step>/units/<unit_id>.sh (in the antiSMASH container when a
 * container engine profile is active). Units read and write <run>/ directly (shared filesystem).
 */

params {
    manifest: Path
    run: String
    step: String
    python: String
    pythonpath: String = ''
    binds: String = ''
}

def pythonTask(String unitId) {
    def env = params.pythonpath ? "export PYTHONPATH='${params.pythonpath}'\n" : ''
    return env + "'${params.python}' -m sanntis task '${params.step}' --run '${params.run}' --unit '${unitId}'"
}

def scriptTask(String unitId) {
    return "bash '${params.run}/${params.step}/units/${unitId}.sh'"
}

def bindOptions() {
    def paths = [params.run] + (params.binds ? params.binds.tokenize(',') : [])
    def engine = workflow.containerEngine
    if (engine == 'singularity' || engine == 'apptainer') {
        return paths.collect { p -> "-B ${p}" }.join(' ')
    }
    if (engine == 'docker' || engine == 'podman') {
        return paths.collect { p -> "-v ${p}:${p}" }.join(' ')
    }
    return ''
}

process TASK_CPU_SMALL {
    label 'cpu_small'
    tag "${unit_id}"

    input:
    val unit_id

    output:
    val unit_id

    script:
    pythonTask(unit_id)
}

process TASK_TRAIN {
    label 'train'
    tag "${unit_id}"

    input:
    val unit_id

    output:
    val unit_id

    script:
    pythonTask(unit_id)
}

process TASK_GPU {
    label 'gpu'
    tag "${unit_id}"
    accelerator 1

    input:
    val unit_id

    output:
    val unit_id

    script:
    pythonTask(unit_id)
}

process TASK_ANTISMASH {
    label 'antismash'
    tag "${unit_id}"
    containerOptions "${bindOptions()}"

    input:
    val unit_id

    output:
    val unit_id

    script:
    "export SANNTIS_CPUS=${task.cpus}\n" + scriptTask(unit_id)
}

workflow {
    rows = channel.fromPath(params.manifest)
        .splitCsv(sep: '\t', header: true)
        .branch { r ->
            cpu_small: r.label == 'cpu_small'
            train: r.label == 'train'
            gpu: r.label == 'gpu'
            antismash: r.label == 'antismash'
            unknown: true
        }
    rows.unknown.subscribe { r -> error("unknown task label '${r.label}' for unit ${r.unit_id}") }
    TASK_CPU_SMALL(rows.cpu_small.map { r -> r.unit_id })
    TASK_TRAIN(rows.train.map { r -> r.unit_id })
    TASK_GPU(rows.gpu.map { r -> r.unit_id })
    TASK_ANTISMASH(rows.antismash.map { r -> r.unit_id })
}
