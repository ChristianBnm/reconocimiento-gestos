
import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API = "/api";

const DATASET_ROLES = [
  { value: "train_source", label: "Entrenamiento" },
  { value: "final_test", label: "Prueba final" },
  { value: "external_eval", label: "Evaluación externa" },
  { value: "unused", label: "No utilizar" },
];

const ROLE_LABELS = {
  train_source: "Entrenamiento",
  final_test: "Prueba final",
  external_eval: "Evaluación externa",
  unused: "No utilizar",
};

const RUN_STATUS_LABELS = {
  pending: "Pendiente",
  running: "En ejecución",
  completed: "Completada",
  failed: "Fallida",
  cancelled: "Cancelada",
};

const ACTIVE_RUN_STATUSES = new Set(["pending", "running"]);

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("es-AR");
}

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });

  const text = await response.text();
  let data = null;

  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : Array.isArray(data?.detail)
          ? data.detail
              .map((item) => item.msg ?? JSON.stringify(item))
              .join("; ")
          : `Error HTTP ${response.status}`;

    throw new Error(`${detail} (HTTP ${response.status})`);
  }

  return data;
}

const INITIAL_FORM = {
  name: "",
  description: "",
  model_name: "",
  datasets: [],
  device: "auto",
  image_size: 224,
  batch_size: 32,
  epochs: 30,
  learning_rate: 0.001,
  weight_decay: 0.0001,
  patience: 5,
  seed: 42,
  freeze_backbone: true,
  train_percent: 70,
  validation_percent: 15,
  test_percent: 15,
};

function App() {
  const [health, setHealth] = useState("checking");
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [sources, setSources] = useState([]);
  const [invalidSources, setInvalidSources] = useState([]);
  const [runs, setRuns] = useState([]);
  const [runHistories, setRunHistories] = useState({});
  const [runsError, setRunsError] = useState("");
  const [runsLoading, setRunsLoading] = useState(false);
  const [launchingId, setLaunchingId] = useState(null);
  const [hoveredActiveRunId, setHoveredActiveRunId] = useState(null);
  const [cancellingRunId, setCancellingRunId] = useState(null);
  const [cancellationRequestedIds, setCancellationRequestedIds] = useState(() => new Set());
  const [error, setError] = useState("");
  const [formError, setFormError] = useState("");
  const [success, setSuccess] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [evaluatingRunId, setEvaluatingRunId] = useState(null);
  const [selectedEvaluation, setSelectedEvaluation] = useState(null);
  const [selectedTestKey, setSelectedTestKey] = useState("original");
  const [evaluationError, setEvaluationError] = useState("");


  const [form, setForm] = useState(INITIAL_FORM);

  const loadRuns = useCallback(async () => {
    try {
      const data = await request("/runs");
      const items = Array.isArray(data) ? data : data?.runs ?? [];
      const ordered = [...items].sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      );

      setRuns(ordered);
      setRunsError("");

      const activeRunIds = new Set(
        ordered
          .filter((run) => ACTIVE_RUN_STATUSES.has(run.status))
          .map((run) => run.run_id),
      );
      setCancellationRequestedIds((current) => {
        const stillPending = [...current].filter((id) => activeRunIds.has(id));
        return new Set(stillPending);
      });

      // history.json se actualiza al terminar cada época. Si todavía no existe,
      // la ejecución simplemente se muestra sin métricas de progreso.
      const terminalRunsWithHistory = new Set(["completed", "failed", "cancelled"]);
      const historyTargets = [
        ...ordered.filter((run) => ACTIVE_RUN_STATUSES.has(run.status)),
        ...ordered.filter((run) => terminalRunsWithHistory.has(run.status)).slice(0, 5),
      ];

      await Promise.all(
        historyTargets.map(async (run) => {
          try {
            const historyData = await request(
              `/runs/${encodeURIComponent(run.run_id)}/history`,
            );
            const history = Array.isArray(historyData)
              ? historyData
              : historyData?.history ?? [];
            setRunHistories((current) => ({
              ...current,
              [run.run_id]: history,
            }));
          } catch {
            // Es normal que aún no exista history.json antes de la primera época.
          }
        }),
      );
    } catch (err) {
      setRunsError(err.message);
    }
  }, []);

  const loadDashboard = useCallback(async () => {
    setError("");

    try {
      const [healthData, experimentData, modelData, sourceData] =
        await Promise.all([
          request("/health"),
          request("/experiments"),
          request("/models"),
          request("/sources"),
        ]);

      setHealth(healthData?.status === "ok" ? "online" : "error");

      setExperiments(
        Array.isArray(experimentData)
          ? experimentData
          : experimentData?.experiments ?? [],
      );

      setModels(
        Array.isArray(modelData) ? modelData : modelData?.models ?? [],
      );

      setSources(
        Array.isArray(sourceData)
          ? sourceData
          : sourceData?.sources ?? [],
      );
      setInvalidSources(
        Array.isArray(sourceData?.invalid_sources)
          ? sourceData.invalid_sources
          : [],
      );

      await loadRuns();
    } catch (err) {
      setHealth("error");
      setError(err.message);
    }
  }, [loadRuns]);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  const hasActiveRun = runs.some((run) => ACTIVE_RUN_STATUSES.has(run.status));

  useEffect(() => {
    if (!hasActiveRun) return undefined;

    const intervalId = window.setInterval(() => {
      loadRuns();
    }, 5000);

    return () => window.clearInterval(intervalId);
  }, [hasActiveRun, loadRuns]);

  function updateForm(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  }

  function updateDatasetAssignment(index, field, value) {
    setForm((current) => ({
      ...current,
      datasets: current.datasets.map((dataset, i) =>
        i === index ? { ...dataset, [field]: value } : dataset
      ),
    }));
  }

  function addDatasetAssignment() {
    const assigned = new Set(
      form.datasets.map((dataset) => dataset.source_name)
    );

    const nextSource = sources.find(
      (source) => !assigned.has(source.name)
    );

    if (!nextSource) {
      setFormError("No hay más datasets disponibles para asignar.");
      return;
    }

    setFormError("");

    setForm((current) => ({
      ...current,
      datasets: [
        ...current.datasets,
        {
          source_name: nextSource.name,
          role: current.datasets.some(
            (dataset) => dataset.role === "train_source"
          )
            ? "final_test"
            : "train_source",
        },
      ],
    }));
  }

  function removeDatasetAssignment(index) {
    setForm((current) => ({
      ...current,
      datasets: current.datasets.filter((_, i) => i !== index),
    }));
  }

  function openForm() {
    setFormError("");
    setSuccess("");

    setForm({
      ...INITIAL_FORM,
      model_name: models[0]?.name ?? "",
      datasets: sources[0]
        ? [
            {
              source_name: sources[0].name,
              role: "train_source",
            },
          ]
        : [],
    });

    setShowForm(true);
  }

  async function createExperiment(event) {
    event.preventDefault();
    setFormError("");
    setSuccess("");

  if (!form.name.trim()) {
    setFormError("Ingresá un nombre para el experimento.");
    return;
  }

  if (!form.model_name) {
    setFormError("Seleccioná un modelo.");
    return;
  }

  if (form.datasets.length === 0) {
    setFormError("Asigná al menos un dataset.");
    return;
  }

  if (!form.datasets.some((dataset) => dataset.role === "train_source")) {
    setFormError("Debe existir al menos un dataset de entrenamiento.");
    return;
  }

  // Evitar que una misma fuente se asigne más de una vez.
  const assignedNames = form.datasets.map(
    (dataset) => dataset.source_name
  );

  if (new Set(assignedNames).size !== assignedNames.length) {
    setFormError("No podés asignar la misma fuente más de una vez.");
    return;
  }

  // Validar porcentajes.
  const trainPercent = Number(form.train_percent);
  const validationPercent = Number(form.validation_percent);
  const testPercent = Number(form.test_percent);

  const percentages = [
    trainPercent,
    validationPercent,
    testPercent,
  ];

  if (
    percentages.some(
      (value) => !Number.isFinite(value) || value <= 0 || value > 100
    )
  ) {
    setFormError("Cada porcentaje debe ser mayor que 0 y no superar 100.");
    return;
  }

  const percentageTotal = percentages.reduce(
    (total, value) => total + value,
    0
  );

  if (Math.abs(percentageTotal - 100) > 0.0001) {
    setFormError("Los porcentajes de entrenamiento, validación y test deben sumar 100%.");
    return;
  }

  // Resolver las fuentes y sus rutas.
  const datasetsPayload = [];

  for (const assignment of form.datasets) {
    const selectedSource = sources.find(
      (source) => source.name === assignment.source_name
    );

    if (!selectedSource?.path) {
      setFormError(
        `La fuente ${assignment.source_name} no tiene una ruta válida.`
      );
      return;
    }

    datasetsPayload.push({
      source_name: selectedSource.name,
      path: selectedSource.path,
      role: assignment.role,
    });
  }

  const trainingAssignment = form.datasets.find(
    (dataset) => dataset.role === "train_source"
  );
  const trainingSource = sources.find(
    (source) => source.name === trainingAssignment?.source_name
  );

  if (!trainingSource) {
    setFormError("No se pudo resolver el vocabulario del dataset de entrenamiento.");
    return;
  }

const payload = {
  name: form.name.trim(),
  description: form.description.trim(),
  model_name: form.model_name,
  class_file: trainingSource.class_file ?? null,
  datasets: datasetsPayload,
  parameters: {
    device: form.device,
    image_size: Number(form.image_size),
    batch_size: Number(form.batch_size),
    epochs: Number(form.epochs),
    learning_rate: Number(form.learning_rate),
    weight_decay: Number(form.weight_decay),
    patience: Number(form.patience),
    seed: Number(form.seed),
    freeze_backbone: form.freeze_backbone,
    train_size: trainPercent / 100,
    val_size: validationPercent / 100,
    test_size: testPercent / 100,
  },
};

    setSaving(true);

    try {
      const created = await request("/experiments", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      setExperiments((current) => [
        created,
        ...current.filter(
          (item) => item.experiment_id !== created.experiment_id,
        ),
      ]);

      setSuccess(`Experimento creado: ${created.experiment_id}`);
      setShowForm(false);

      setForm({
        ...INITIAL_FORM,
        model_name: models[0]?.name ?? "",
        datasets: sources[0]
          ? [
          {
            source_name: sources[0].name,
            role: "train_source",
          },
        ]
      : [],
    });

    } catch (err) {
      setFormError(err.message);
    } finally {
      setSaving(false);
    }

  }

  async function refreshRuns() {
    setRunsLoading(true);
    try {
      await loadRuns();
    } finally {
      setRunsLoading(false);
    }
  }

  async function startRun(experiment) {
    const experimentId = experiment?.experiment_id;
    if (!experimentId) return;

    if (hasActiveRun) {
      setError("Ya hay un entrenamiento en curso. Esperá a que finalice antes de iniciar otro.");
      return;
    }

    setLaunchingId(experimentId);
    setError("");
    setSuccess("");

    try {
      const created = await request(
        `/experiments/${encodeURIComponent(experimentId)}/runs`,
        { method: "POST" },
      );

      setRuns((current) => [
        created,
        ...current.filter((run) => run.run_id !== created.run_id),
      ]);
      setSuccess(`Entrenamiento iniciado: ${created.run_id}`);
      await loadRuns();
    } catch (err) {
      setError(`No se pudo iniciar ${experimentId}: ${err.message}`);
    } finally {
      setLaunchingId(null);
    }
  }

  async function evaluateRun(run) {
  if (!run?.run_id || run.status !== "completed") return;

  setEvaluatingRunId(run.run_id);
  setEvaluationError("");
  setError("");
  setSuccess("");

  try {
    // Ejecutar la evaluación mediante FastAPI.
    await request(
      `/runs/${encodeURIComponent(run.run_id)}/evaluate`,
      { method: "POST" }
    );

    // Recuperar las métricas y la matriz de confusión generadas.
    const results = await request(
      `/runs/${encodeURIComponent(run.run_id)}/results`
    );

    setSelectedEvaluation({
      runId: run.run_id,
      results,
    });

    setSuccess(`Evaluación completada: ${run.run_id}`);

    // Llevar al usuario a la sección de resultados.
    document.getElementById("evaluation-results")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  } catch (err) {
    setEvaluationError(
      `No se pudo evaluar ${run.run_id}: ${err.message}`
    );
  } finally {
    setEvaluatingRunId(null);
  }
}

  async function cancelRun(run) {
    if (!run?.run_id || !ACTIVE_RUN_STATUSES.has(run.status)) return;

    const confirmed = window.confirm(
      `¿Detener el entrenamiento ${run.run_id}?\n\n` +
        "La ejecución quedará marcada como cancelada. Se conservará el historial de épocas ya completadas."
    );
    if (!confirmed) return;

    setCancellingRunId(run.run_id);
    setError("");
    setSuccess("");
    try {
      await request(`/runs/${encodeURIComponent(run.run_id)}/cancel`, {
        method: "POST",
      });
      setCancellationRequestedIds((current) => {
        const next = new Set(current);
        next.add(run.run_id);
        return next;
      });
      setSuccess(
        `Solicitud de detención enviada para ${run.run_id}. ` +
          "Se marcará como cancelada cuando finalice el paso de entrenamiento en curso.",
      );
      await loadRuns();
    } catch (err) {
      setError(`No se pudo detener ${run.run_id}: ${err.message}`);
    } finally {
      setCancellingRunId(null);
      setHoveredActiveRunId(null);
    }
  }

  async function deleteExperiment(experiment) {
    const { experiment_id, name } = experiment;

    if (runsError) {
      setError("No se puede verificar el historial de ejecuciones. Actualizá los datos e intentá nuevamente.");
      return;
    }

    if (runs.some((run) => run.experiment_id === experiment_id)) {
      setError(
        `No se puede eliminar ${experiment_id}: tiene ejecuciones asociadas. El historial se conserva.`,
      );
      return;
    }

    const confirmed = window.confirm(
      `¿Querés eliminar el experimento "${name}" (${experiment_id})?\n\n` +
        "Solo se eliminará la configuración. Los experimentos con ejecuciones asociadas no se pueden eliminar para conservar su historial."
    );

    if (!confirmed) return;

    setDeletingId(experiment_id);
    setError("");
    setSuccess("");

    try {
      await request(`/experiments/${encodeURIComponent(experiment_id)}`, {
        method: "DELETE",
      });

      setExperiments((current) =>
        current.filter((item) => item.experiment_id !== experiment_id)
      );

      setSuccess(`Experimento ${experiment_id} eliminado correctamente.`);
    } catch (err) {
      setError(`No se pudo eliminar ${experiment_id}: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">LSA</div>
          <div>
            <strong>Gesture Lab</strong>
            <span>Research workspace</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav>
          <a className="nav-item active" href="#dashboard">
            ▦ Dashboard
          </a>
          <a className="nav-item" href="#experiments">
            ◈ Experimentos
          </a>
          <a className="nav-item" href="#runs">
            ▷ Ejecuciones
          </a>
          <a className="nav-item" href="#resources">
            ⌘ Recursos
          </a>
        </nav>

        <div className="sidebar-footer">
          <span className={`status-dot ${health}`} />
          API{" "}
          {health === "online"
            ? "conectada"
            : health === "checking"
              ? "verificando..."
              : "sin conexión"}
        </div>
      </aside>

      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div>
            <span className="eyebrow">
              TESIS · RECONOCIMIENTO DE GESTOS
            </span>
            <h1>Panel de investigación</h1>
            <p>
              Administrá experimentos y analizá ejecuciones de modelos.
            </p>
          </div>

          <div className="api-badge">
            <span className={`status-dot ${health}`} />
            {health === "online"
              ? "Sistema operativo"
              : health === "checking"
                ? "Conectando"
                : "Error de conexión"}
          </div>
        </header>

        {error && (
          <div className="error-banner">
            No se pudieron cargar todos los datos de la API: {error}
            <button type="button" onClick={loadDashboard}>
              Reintentar
            </button>
          </div>
        )}

        {success && <div className="success-banner">{success}</div>}

        <section className="stats-grid">
          <article className="stat-card">
            <span className="stat-icon purple">◈</span>
            <span className="stat-label">Experimentos</span>
            <strong>{experiments.length}</strong>
            <span className="stat-caption">Registrados en el sistema</span>
          </article>

          <article className="stat-card">
            <span className="stat-icon blue">⌘</span>
            <span className="stat-label">Modelos disponibles</span>
            <strong>{models.length}</strong>
            <span className="stat-caption">Opciones de arquitectura</span>
          </article>

          <article className="stat-card">
            <span className="stat-icon green">▤</span>
            <span className="stat-label">Fuentes de datos</span>
            <strong>{sources.length}</strong>
            <span className="stat-caption">Fuentes configuradas</span>
          </article>

          <article className="stat-card">
            <span className="stat-icon orange">◉</span>
            <span className="stat-label">API</span>
            <strong className="api-stat">
              {health === "online"
                ? "Online"
                : health === "checking"
                  ? "..."
                  : "Error"}
            </strong>
            <span className="stat-caption">FastAPI · REST</span>
          </article>
        </section>

        <section className="content-panel" id="experiments">
          <div className="panel-heading">
            <div>
              <h2>Experimentos</h2>
              <p>Configuraciones registradas para la investigación.</p>
            </div>

            <div className="panel-actions">
              <span className="count-badge">
                {experiments.length} total
              </span>
              <button
                type="button"
                className="primary-button"
                onClick={openForm}
              >
                + Nuevo experimento
              </button>
            </div>
          </div>

          {showForm && (
            <form className="experiment-form" onSubmit={createExperiment}>
              <div className="form-heading">
                <div>
                  <h3>Crear experimento</h3>
                  <p>Definí el modelo y la fuente de datos inicial.</p>
                </div>

                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setShowForm(false);
                    setFormError("");
                  }}
                  disabled={saving}
                >
                  Cancelar
                </button>
              </div>

              <div className="form-grid">
                <label>
                  Nombre *
                  <input
                    name="name"
                    value={form.name}
                    onChange={updateForm}
                    placeholder="Ej. ViT Tiny - Dataset original"
                    required
                    maxLength={120}
                  />
                </label>

                <label>
                  Modelo *
                  <select
                    name="model_name"
                    value={form.model_name}
                    onChange={updateForm}
                    required
                  >
                    <option value="">Seleccionar modelo</option>
                    {models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name} — {model.family}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="form-full">
                  <h3>Datasets</h3>
                  <p>
                    Seleccioná las fuentes que participarán en el experimento
                    y asignales un propósito.
                  </p>

                  {form.datasets.map((dataset, index) => (
                    <div
                      className="form-grid"
                      key={`${dataset.source_name}-${index}`}
                    >
                      <label>
                        Dataset *
                        <select
                          value={dataset.source_name}
                          onChange={(event) =>
                            updateDatasetAssignment(
                              index,
                              "source_name",
                              event.target.value
                            )
                          }
                          required
                        >
                          {sources.map((source) => (
                            <option
                              key={source.name}
                              value={source.name}
                              disabled={form.datasets.some(
                                (item, i) =>
                                  i !== index &&
                                  item.source_name === source.name
                              )}
                            >
                              {source.name} — {source.num_classes} clases · {source.num_images} imágenes
                            </option>
                          ))}
                        </select>
                      </label>

                      <label>
                        Rol del dataset *
                        <select
                          value={dataset.role}
                          onChange={(event) =>
                            updateDatasetAssignment(
                              index,
                              "role",
                              event.target.value
                            )
                          }
                          required
                        >
                          {DATASET_ROLES.map((role) => (
                            <option key={role.value} value={role.value}>
                              {role.label}
                            </option>
                          ))}
                        </select>
                      </label>

                      <div className="form-full">
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => removeDatasetAssignment(index)}
                        >
                          Quitar dataset
                        </button>
                      </div>
                    </div>
                  ))}

                  <button
                    type="button"
                    className="secondary-button"
                    onClick={addDatasetAssignment}
                    disabled={form.datasets.length >= sources.length}
                  >
                    + Agregar dataset
                  </button>

                  <p className="form-note">
                    Debe existir al menos un dataset asignado a Entrenamiento.
                    Cada fuente puede asignarse una sola vez.
                  </p>
                </div>

                                <div className="form-full">
                  <h3>Distribución del dataset de entrenamiento</h3>

                  <p>
                    Definí cómo se dividirán los datos de las fuentes
                    asignadas al entrenamiento.
                  </p>

                  <div className="form-grid">
                    <label>
                      Entrenamiento (%)
                      <input
                        type="number"
                        name="train_percent"
                        min="1"
                        max="100"
                        step="any"
                        value={form.train_percent}
                        onChange={updateForm}
                        required
                      />
                    </label>

                    <label>
                      Validación (%)
                      <input
                        type="number"
                        name="validation_percent"
                        min="1"
                        max="100"
                        step="any"
                        value={form.validation_percent}
                        onChange={updateForm}
                        required
                      />
                    </label>

                    <label>
                      Test (%)
                      <input
                        type="number"
                        name="test_percent"
                        min="1"
                        max="100"
                        step="any"
                        value={form.test_percent}
                        onChange={updateForm}
                        required
                      />
                    </label>
                  </div>

                  <p className="form-note">
                    Total:{" "}
                    {(
                      Number(form.train_percent || 0) +
                      Number(form.validation_percent || 0) +
                      Number(form.test_percent || 0)
                    ).toFixed(2)}
                    % — Los porcentajes deben sumar exactamente 100%.
                  </p>

                  <p className="form-note">
                    La validación y el test interno se generan a partir de
                    las fuentes asignadas a entrenamiento. Una fuente con
                    rol Prueba final se conserva como conjunto independiente.
                  </p>
                </div>

                <label>
                  Dispositivo de ejecución *
                  <select
                    name="device"
                    value={form.device}
                    onChange={updateForm}
                    required
                  >
                    <option value="auto">Automático</option>
                    <option value="cpu">CPU</option>
                    <option value="cuda">GPU NVIDIA (CUDA)</option>
                  </select>
                </label>

                <label>
                  Tamaño de imagen *
                  <input
                    type="number"
                    name="image_size"
                    min="1"
                    value={form.image_size}
                    onChange={updateForm}
                    required
                  />
                </label>

                <label>
                  Tamaño del batch *
                  <input
                    type="number"
                    name="batch_size"
                    min="1"
                    value={form.batch_size}
                    onChange={updateForm}
                    required
                  />
                </label>

                <label>
                  Épocas *
                  <input
                    type="number"
                    name="epochs"
                    min="1"
                    value={form.epochs}
                    onChange={updateForm}
                    required
                  />
                </label>

                <label>
                  Learning rate *
                  <input
                    type="number"
                    name="learning_rate"
                    min="0.0000001"
                    step="any"
                    value={form.learning_rate}
                    onChange={updateForm}
                    required
                  />
                </label>

                <label>
                  Weight decay *
                  <input
                    type="number"
                    name="weight_decay"
                    min="0"
                    step="any"
                    value={form.weight_decay}
                    onChange={updateForm}
                    required
                  />
                </label>

                <label>
                  Paciencia (early stopping) *
                  <input
                    type="number"
                    name="patience"
                    min="1"
                    value={form.patience}
                    onChange={updateForm}
                    required
                  />
                </label>

                <label>
                  Semilla aleatoria *
                  <input
                    type="number"
                    name="seed"
                    min="0"
                    value={form.seed}
                    onChange={updateForm}
                    required
                  />
                </label>

                <label className="form-full">
                  <input
                    type="checkbox"
                    name="freeze_backbone"
                    checked={form.freeze_backbone}
                    onChange={(event) =>
                      setForm((current) => ({
                        ...current,
                        freeze_backbone: event.target.checked,
                      }))
                    }
                  />
                  Congelar backbone durante el entrenamiento
                </label>

                <label className="form-f-ull">
                  Descripción
                  <textarea
                    name="description"
                    value={form.description}
                    onChange={updateForm}
                    placeholder="Objetivo o hipótesis del experimento"
                    rows={3}
                  />
                </label>
              </div>

              <div className="form-note">
                La ID se genera automáticamente. Para que el experimento
                pueda crearse, al menos una fuente debe tener el rol
                Entrenamiento.
              </div>

              {formError && (
                <div className="error-banner">{formError}</div>
              )}

              <div className="form-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setShowForm(false);
                    setFormError("");
                  }}
                  disabled={saving}
                >
                  Cancelar
                </button>

                <button
                  type="submit"
                  className="primary-button"
                  disabled={
                    saving || models.length === 0 || sources.length === 0
                  }
                >
                  {saving ? "Creando..." : "Crear experimento"}
                </button>
              </div>
            </form>
          )}

          {/* Tabla de experimentos: encabezado y filas tienen seis columnas. */}
          {experiments.length > 0 ? (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Experimento</th>
                    <th>Modelo</th>
                    <th>Fuente de datos</th>
                    <th>Rol</th>
                    <th>Descripción</th>
                    <th>ID</th>
                    <th>Acciones</th>
                  </tr>
                </thead>

                <tbody>
                  {experiments.map((experiment, index) => (
                    <tr key={experiment.experiment_id ?? index}>
                      <td>
                        <strong>{experiment.name ?? "Sin nombre"}</strong>
                      </td>

                      <td>{experiment.model_name ?? "—"}</td>

                      <td>
                        {experiment.datasets?.length
                          ? experiment.datasets.map((dataset, datasetIndex) => (
                              <div
                                key={`${dataset.source_name}-${datasetIndex}`}
                              >
                                {dataset.source_name ?? "—"}
                              </div>
                            ))
                          : "—"}
                      </td>

                      <td>
                        {experiment.datasets?.length
                          ? experiment.datasets.map((dataset, datasetIndex) => (
                              <div
                                key={`${dataset.source_name}-${datasetIndex}`}
                              >
                                {ROLE_LABELS[dataset.role] ??
                                  dataset.role ??
                                  "—"}
                              </div>
                            ))
                          : "—"}
                      </td>
                      <td>{experiment.description || "—"}</td>
                      <td>
                        <code>
                          {experiment.experiment_id ?? "Sin ID"}
                        </code>
                      </td>
                      <td>
                        <div className="panel-actions">
                          {(() => {
                            const activeRun = runs.find(
                              (run) =>
                                run.experiment_id === experiment.experiment_id &&
                                ACTIVE_RUN_STATUSES.has(run.status),
                            );
                            const isHovered = activeRun && hoveredActiveRunId === activeRun.run_id;
                            const isCancelling = activeRun && (
                              cancellingRunId === activeRun.run_id ||
                              cancellationRequestedIds.has(activeRun.run_id)
                            );
                            return (
                              <button
                                type="button"
                                className={activeRun ? "secondary-button" : "primary-button"}
                                disabled={
                                  !experiment.experiment_id ||
                                  (hasActiveRun && !activeRun) ||
                                  launchingId !== null ||
                                  Boolean(isCancelling)
                                }
                                onMouseEnter={() => activeRun && setHoveredActiveRunId(activeRun.run_id)}
                                onMouseLeave={() => setHoveredActiveRunId(null)}
                                onClick={() => activeRun ? cancelRun(activeRun) : startRun(experiment)}
                                title={activeRun ? "Detener este entrenamiento" : hasActiveRun ? "Hay otro entrenamiento en curso" : "Iniciar entrenamiento con esta configuración"}
                                style={activeRun && isHovered ? { background: "#b91c1c", color: "#ffffff", borderColor: "#b91c1c" } : undefined}
                              >
                                {launchingId === experiment.experiment_id
                                  ? "Iniciando..."
                                  : isCancelling
                                    ? "Deteniendo..."
                                    : activeRun
                                      ? isHovered ? "Detener" : "Ejecución activa"
                                      : hasActiveRun
                                        ? "Ejecución activa"
                                        : "Ejecutar"}
                              </button>
                            );
                          })()}
                          <button
                            type="button"
                            className="secondary-button"
                            disabled={
                              !experiment.experiment_id ||
                              deletingId !== null ||
                              Boolean(runsError) ||
                              runs.some(
                                (run) => run.experiment_id === experiment.experiment_id,
                              )
                            }
                            title={
                              runsError
                                ? "No se puede verificar el historial de ejecuciones."
                                : runs.some(
                                    (run) => run.experiment_id === experiment.experiment_id,
                                  )
                                  ? "No se puede eliminar: el experimento tiene ejecuciones asociadas y su historial se conserva."
                                  : "Eliminar experimento"
                            }
                            onClick={() => deleteExperiment(experiment)}
                          >
                            {deletingId === experiment.experiment_id
                              ? "Eliminando..."
                              : runsError
                                ? "Historial no disponible"
                                : runs.some(
                                    (run) => run.experiment_id === experiment.experiment_id,
                                  )
                                  ? "Con historial"
                                  : "Eliminar"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              <h3>
                {error
                  ? "No se pudieron cargar los experimentos"
                  : "Todavía no hay experimentos"}
              </h3>
              <p>
                {error
                  ? "Revisá la conexión con la API y los registros del backend."
                  : "Creá tu primer experimento para comenzar a registrar configuraciones."}
              </p>
            </div>
          )}
        </section>

        <section className="content-panel" id="runs">
          <div className="panel-heading">
            <div>
              <h2>Ejecuciones recientes</h2>
              <p>Seguimiento de los entrenamientos realizados.</p>
            </div>
            <div className="panel-actions">
              <span className="count-badge">{runs.length} total</span>
              <button
                type="button"
                className="secondary-button"
                onClick={refreshRuns}
                disabled={runsLoading}
              >
                {runsLoading ? "Actualizando..." : "Actualizar"}
              </button>
            </div>
          </div>

          {runsError && (
            <div className="error-banner">
              No se pudieron cargar las ejecuciones: {runsError}
            </div>
          )}

          {runs.length > 0 ? (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Estado</th>
                    <th>Experimento</th>
                    <th>Modelo</th>
                    <th>Progreso</th>
                    <th>Creada</th>
                    <th>ID de ejecución</th>
                    <th>Acciones</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => {
                    const experiment = experiments.find(
                      (item) => item.experiment_id === run.experiment_id,
                    );
                    const history = runHistories[run.run_id] ?? [];
                    const latestEpoch = history[history.length - 1];
                    const expectedEpochs = Number(
                      experiment?.parameters?.epochs ?? 0,
                    );

                    return (
                      <tr key={run.run_id}>
                        <td>
                          <strong>{RUN_STATUS_LABELS[run.status] ?? run.status}</strong>
                          {run.error_message && (
                            <div className="resource-detail">{run.error_message}</div>
                          )}
                        </td>
                        <td>
                          {experiment?.name ?? run.experiment_id}
                          <div className="resource-detail">{run.experiment_id}</div>
                        </td>
                        <td>{experiment?.model_name ?? "—"}</td>
                        <td>
                          {latestEpoch
                            ? `Época ${latestEpoch.epoch}${expectedEpochs ? ` / ${expectedEpochs}` : ""}`
                            : ACTIVE_RUN_STATUSES.has(run.status)
                              ? "Preparando entrenamiento..."
                              : "—"}
                          {latestEpoch && (
                            <div className="resource-detail">
                              Val. accuracy: {(latestEpoch.val_accuracy * 100).toFixed(2)}%
                            </div>
                          )}
                        </td>
                        <td>{formatDate(run.created_at)}</td>
                        <td>
                          <code>{run.run_id}</code>
                        </td>

                        <td>
                          {run.status === "completed" ? (
                            <button
                              type="button"
                              className="primary-button"
                              onClick={() => evaluateRun(run)}
                              disabled={evaluatingRunId !== null}
                            >
                              {evaluatingRunId === run.run_id
                              ? "Evaluando..."
                              : "Evaluar"}
                            </button>
                          ) : (
                            <span className="resource-detail">
                              No disponible
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state compact">
              <p>
                {runsError
                  ? "Revisá la conexión con la API para consultar las ejecuciones."
                  : "Todavía no hay ejecuciones. Usá Ejecutar en un experimento para iniciar el primer entrenamiento."}
              </p>
            </div>
          )}
          {hasActiveRun && (
            <p className="form-note">
              Hay un entrenamiento activo. Los estados y las métricas se actualizan automáticamente cada 5 segundos.
            </p>
          )}
        </section>

        {evaluationError && (
          <div className="error-banner" role="alert">
            {evaluationError}
          </div>
        )}

        {selectedEvaluation && (() => {
            const result = selectedEvaluation.results;

            // Evaluaciones agrupadas por fuente de datos.
            const tests = result?.tests ?? {};
            const testKeys = Object.keys(tests);

            // Usar la fuente seleccionada si existe.
            // Si no existe, seleccionar la primera disponible.
            const activeTestKey = tests[selectedTestKey]
                ? selectedTestKey
                : testKeys[0];

            const test = activeTestKey ? tests[activeTestKey] : null;
            const metrics = test?.metrics ?? {};
            const matrix = test?.confusion_matrix ?? [];
            const classNames = Object.keys(test?.per_class ?? {});

            const resources = [
                ["Fuente evaluada", activeTestKey ?? "—"],
                ["Modelo", result?.model ?? "—"],
                ["Arquitectura", result?.architecture ?? "—"],
                ["Dispositivo solicitado", result?.requested_device ?? "—"],
                ["Dispositivo utilizado", result?.device ?? "—"],
                ["Checkpoint", result?.checkpoint_path ?? "—"],
                ["Partición evaluada", test?.partition ?? "—"],
                ["Muestras evaluadas", test?.num_samples ?? "—"],
            ];

            const formatMetric = (value) => {
                if (value == null || !Number.isFinite(Number(value))) {
                    return "—";
                }

                return `${(Number(value) * 100).toFixed(2)}%`;
            };

            const metricCards = [
                {
                    label: "Accuracy",
                    value: metrics.accuracy,
                },
                {
                    label: "F1 macro",
                    value: metrics.f1_macro ?? metrics.macro_f1,
                },
                {
                    label: "Precisión macro",
                    value: metrics.precision_macro,
                },
                {
                    label: "Recall macro",
                    value: metrics.recall_macro,
                },
            ];

            return (
                <section
                    className="content-panel"
                    id="evaluation-results"
                >
                    <div className="panel-heading">
                        <div>
                            <h2>Resultados de evaluación</h2>
                            <p>
                                Ejecución: {selectedEvaluation.runId}
                            </p>
                        </div>

                        <button
                            type="button"
                            className="secondary-button"
                            onClick={() => {
                                setSelectedEvaluation(null);
                                setSelectedTestKey("original");
                            }}
                        >
                            Cerrar
                        </button>
                    </div>

                    {testKeys.length > 0 ? (
                        <>
                            <div className="evaluation-details">
                                <label htmlFor="evaluation-source">
                                    <strong>Conjunto evaluado</strong>
                                </label>

                                <select
                                    id="evaluation-source"
                                    value={activeTestKey ?? ""}
                                    onChange={(event) =>
                                        setSelectedTestKey(event.target.value)
                                    }
                                >
                                    {testKeys.map((key) => (
                                        <option key={key} value={key}>
                                            {key}
                                        </option>
                                    ))}
                                </select>

                                <p className="resource-detail">
                                    Partición: {test?.partition ?? "—"} ·
                                    {" "}Muestras: {test?.num_samples ?? "—"}
                                </p>
                            </div>

                            <div className="evaluation-grid">
                                {metricCards.map((metric) => (
                                    <article
                                        className="stat-card"
                                        key={metric.label}
                                    >
                                        <span className="stat-label">
                                            {metric.label}
                                        </span>

                                        <strong>
                                            {formatMetric(metric.value)}
                                        </strong>
                                    </article>
                                ))}
                            </div>

                            <div className="evaluation-details">
                                <h3>Métricas adicionales</h3>

                                <p>
                                    <strong>Loss:</strong>{" "}
                                    {Number.isFinite(Number(metrics.loss)) &&
                                    metrics.loss != null
                                        ? Number(metrics.loss).toFixed(5)
                                        : "—"}
                                </p>

                                <p>
                                    <strong>Clases:</strong>{" "}
                                    {result?.num_classes ?? classNames.length ?? "—"}
                                </p>

                                <p>
                                    <strong>Dispositivo:</strong>{" "}
                                    {result?.device ?? "—"}
                                </p>

                                <h3>Matriz de confusión</h3>

                                <p className="resource-detail">
                                    Filas: clase real · Columnas: clase predicha.
                                </p>

                                {Array.isArray(matrix) && matrix.length > 0 ? (
                                    <div className="confusion-wrapper">
                                        <table className="confusion-matrix">
                                            <thead>
                                                <tr>
                                                    <th className="matrix-axis-label">
                                                        Real ↓ / Pred. →
                                                    </th>

                                                    {matrix.map((_, index) => (
                                                        <th
                                                            key={index}
                                                            title={`Clase predicha: ${
                                                                classNames[index] ?? index
                                                            }`}
                                                        >
                                                            {classNames[index] ?? index}
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>

                                            <tbody>
                                                {matrix.map((row, rowIndex) => (
                                                    <tr key={rowIndex}>
                                                        <th
                                                            scope="row"
                                                            title={`Clase real: ${
                                                                classNames[rowIndex] ?? rowIndex
                                                            }`}
                                                        >
                                                            {classNames[rowIndex] ?? rowIndex}
                                                        </th>

                                                        {row.map((value, colIndex) => (
                                                            <td
                                                                key={colIndex}
                                                                className={
                                                                    rowIndex === colIndex
                                                                        ? "matrix-diagonal"
                                                                        : value > 0
                                                                          ? "matrix-error"
                                                                          : ""
                                                                }
                                                                title={`Real: ${
                                                                    classNames[rowIndex] ?? rowIndex
                                                                }; predicha: ${
                                                                    classNames[colIndex] ?? colIndex
                                                                }; cantidad: ${value}`}
                                                            >
                                                                {value}
                                                            </td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : (
                                    <p className="resource-detail">
                                        La API no devolvió una matriz de confusión
                                        en el formato esperado para esta fuente.
                                    </p>
                                )}

                                <h3>Configuración y recursos utilizados</h3>

                                <div className="evaluation-resources">
                                    {resources.map(([label, value]) => (
                                        <div
                                            className="evaluation-resource"
                                            key={label}
                                        >
                                            <span>{label}</span>
                                            <strong>{String(value)}</strong>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </>
                    ) : (
                        <p className="resource-detail">
                            La API no devolvió evaluaciones agrupadas por fuente
                            en result.tests. Revisá la estructura de la respuesta
                            del endpoint /results.
                        </p>
                    )}
                </section>
            );
        })()}


        <section className="content-panel" id="resources">
          <div className="panel-heading">
            <div>
              <h2>Recursos disponibles</h2>
              <p>Modelos y datasets detectados por el backend al iniciar.</p>
            </div>
          </div>

          <div className="resource-grid">
            <div className="resource-card">
              <span>MODELOS</span>
              <strong>{models.length}</strong>
              {models.length > 0 ? (
                models.map((model) => (
                  <p key={model.name}>
                    {model.name}{" "}
                    <span className="resource-detail">{model.family}</span>
                  </p>
                ))
              ) : (
                <p>No se encontraron modelos.</p>
              )}
            </div>

            <div className="resource-card">
              <span>DATASETS VÁLIDOS</span>
              <strong>{sources.length}</strong>
              {sources.length > 0 ? (
                sources.map((source) => (
                  <p key={source.name}>
                    <strong>{source.name}</strong>
                    <span className="resource-detail">
                      {source.num_classes} clases · {source.num_images} imágenes
                    </span>
                    <span className="resource-detail">
                      {source.description}
                    </span>
                    <span className="resource-detail">
                      Clases: {(source.classes ?? []).join(", ")}
                    </span>
                  </p>
                ))
              ) : (
                <p>No se detectaron datasets válidos.</p>
              )}
            </div>

            {invalidSources.length > 0 && (
              <div className="resource-card resource-card-warning">
                <span>CARPETAS RECHAZADAS</span>
                <strong>{invalidSources.length}</strong>
                {invalidSources.map((source) => (
                  <p key={`${source.name}-${source.path}`}>
                    <strong>{source.name}</strong>
                    <span className="resource-detail">{source.path}</span>
                    <span className="resource-detail">{source.reason}</span>
                  </p>
                ))}
              </div>
            )}
          </div>
        </section>

        <footer>
          <span>Gesture Lab · Proyecto de tesis</span>
          <span>React + FastAPI + Docker</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
