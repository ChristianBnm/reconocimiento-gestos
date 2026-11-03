
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
  source_name: "",
  role: "train_source",
  device: "auto",
  image_size: 224,
  batch_size: 32,
  epochs: 30,
  learning_rate: 0.001,
  weight_decay: 0.0001,
  patience: 5,
  seed: 42,
  freeze_backbone: true,
};

function App() {
  const [health, setHealth] = useState("checking");
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [sources, setSources] = useState([]);
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

  function openForm() {
    setFormError("");
    setSuccess("");
    setForm({
      ...INITIAL_FORM,
      model_name: models[0]?.name ?? "",
      source_name: sources[0]?.name ?? "",
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

    if (!form.source_name) {
      setFormError("Seleccioná una fuente de datos.");
      return;
    }

    // Esta primera versión admite una única fuente por experimento.
    // La API exige que exista al menos una fuente de entrenamiento.
    if (form.role !== "train_source") {
      setFormError(
        "En esta versión, la fuente debe tener el rol Entrenamiento.",
      );
      return;
    }

    const selectedSource = sources.find(
      (source) => source.name === form.source_name,
    );

    if (!selectedSource?.path) {
      setFormError("La fuente seleccionada no tiene una ruta válida.");
      return;
    }

    const payload = {
      name: form.name.trim(),
      description: form.description.trim(),
      model_name: form.model_name,
      datasets: [
        {
          source_name: selectedSource.name,
          path: selectedSource.path,
          role: form.role,
        },
      ],
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
        source_name: sources[0]?.name ?? "",
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

                <label>
                  Fuente de datos *
                  <select
                    name="source_name"
                    value={form.source_name}
                    onChange={updateForm}
                    required
                  >
                    <option value="">Seleccionar fuente</option>
                    {sources.map((source) => (
                      <option key={source.name} value={source.name}>
                        {source.name} — {source.description}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  Rol de la fuente *
                  <select
                    name="role"
                    value={form.role}
                    onChange={updateForm}
                    required
                  >
                    {DATASET_ROLES.map((role) => (
                      <option key={role.value} value={role.value}>
                        {role.label}
                      </option>
                    ))}
                  </select>
                </label>

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

          // La API devuelve las evaluaciones agrupadas por fuente.
          const tests = result?.tests ?? {};

          // Priorizar la fuente original; utilizar otra disponible
          // como alternativa.
          const test =
            tests.original ??
            Object.values(tests)[0] ??
            null;

          const metrics = test?.metrics ?? {};

          const matrix = test?.confusion_matrix ?? [];
          const classNames = Object.keys(test?.per_class ?? {});
          const resources = [
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

            const number = Number(value);
            return `${(number * 100).toFixed(2)}%`;
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
                  onClick={() => setSelectedEvaluation(null)}
                >
                  Cerrar
                </button>
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
                  {Number.isFinite(Number(metrics.loss))
                    ? Number(metrics.loss).toFixed(5)
                    : "—"}
                </p>

                <p>
                  <strong>Clases:</strong>{" "}
                  {result?.num_classes ?? "—"}
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
                          <th className="matrix-axis-label">Real ↓ / Pred. →</th>
                          {matrix.map((_, index) => (
                            <th key={index} title={`Clase predicha: ${classNames[index] ?? index}`}>
                              {classNames[index] ?? index}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {matrix.map((row, rowIndex) => (
                          <tr key={rowIndex}>
                            <th scope="row" title={`Clase real: ${classNames[rowIndex] ?? rowIndex}`}>
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
                                title={`Real: ${classNames[rowIndex] ?? rowIndex}; predicha: ${classNames[colIndex] ?? colIndex}; cantidad: ${value}`}
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
                    La API no devolvió una matriz de confusión en el formato esperado.
                  </p>
                )}

                <h3>Configuración y recursos utilizados</h3>
                <div className="evaluation-resources">
                  {resources.map(([label, value]) => (
                    <div className="evaluation-resource" key={label}>
                      <span>{label}</span>
                      <strong>{String(value)}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          );
        })()}


        <section className="content-panel" id="resources">
          <div className="panel-heading">
            <div>
              <h2>Recursos disponibles</h2>
              <p>Modelos y fuentes de datos expuestos por la API.</p>
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
              <span>FUENTES DE DATOS</span>
              <strong>{sources.length}</strong>

              {sources.length > 0 ? (
                sources.map((source) => (
                  <p key={source.name}>
                    {source.name}
                    <span className="resource-detail">
                      {source.description}
                    </span>
                  </p>
                ))
              ) : (
                <p>No se encontraron fuentes.</p>
              )}
            </div>
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
