import { Camera, Download, ShieldAlert, Upload } from 'lucide-react'
import { useRef, useState } from 'react'

import { Badge, ErrorBanner, Loading, Notice, Panel, StatCard } from '@/components/ui'
import { useAction, useAsync } from '@/hooks/useApi'
import { getVisionCategories, inspectImage } from '@/lib/endpoints'
import { storageUrl } from '@/lib/api'
import type { VisualInspectionResponse } from '@/types/api'

export default function VisionPage() {
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [category, setCategory] = useState('')
  const categories = useAsync(() => getVisionCategories(), [])
  const action = useAction<[File, string | undefined], VisualInspectionResponse>(inspectImage)
  const result = action.result

  const submit = () => {
    if (file) action.run(file, category || undefined)
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">Visual Inspection</h1>
        <p className="mt-1 text-[13px] text-slate-400">
          PatchCore patch-memory detector over MVTec AD. Upload a product image to obtain a
          localisation heatmap and overlay.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-5">
        <Panel className="lg:col-span-2" title="Upload" subtitle="PNG, JPG, BMP, WEBP or TIFF">
          <div className="space-y-4">
            <div
              onClick={() => fileRef.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault()
                const dropped = event.dataTransfer.files?.[0]
                if (dropped) setFile(dropped)
              }}
              className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-white/10 px-6 py-12 text-center transition hover:border-cyan-brand/40 hover:bg-white/[0.02]"
            >
              <Upload className="h-7 w-7 text-slate-600" />
              <p className="text-[13px] text-slate-400">
                {file ? file.name : 'Drop an image or click to browse'}
              </p>
              {file && <p className="mono text-[11px] text-slate-600">{(file.size / 1024).toFixed(0)} KB</p>}
              <input
                ref={fileRef}
                type="file"
                accept="image/png,image/jpeg,image/bmp,image/webp,image/tiff"
                className="hidden"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </div>

            <div>
              <label className="label" htmlFor="category">
                MVTec category
              </label>
              <select
                id="category"
                value={category}
                onChange={(event) => setCategory(event.target.value)}
                className="input"
              >
                <option value="">Detect automatically</option>
                {(categories.data?.expected_categories ?? []).map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </div>

            <button className="btn-primary w-full" disabled={!file || action.pending} onClick={submit}>
              <Camera className="h-4 w-4" />
              {action.pending ? 'Inspecting...' : 'Inspect image'}
            </button>

            {categories.data && (
              <div
                className={`rounded-lg border p-3 text-[11px] leading-relaxed ${
                  categories.data.loaded
                    ? 'border-risk-healthy/25 bg-risk-healthy/[0.06] text-emerald-200'
                    : 'border-amber-500/30 bg-amber-500/[0.07] text-amber-200'
                }`}
              >
                <p className="mb-1 flex items-center gap-1.5 font-semibold">
                  {categories.data.loaded ? (
                    'Model trained'
                  ) : (
                    <>
                      <ShieldAlert className="h-3.5 w-3.5" /> Model not trained
                    </>
                  )}
                </p>
                {categories.data.detail}
              </div>
            )}
          </div>
        </Panel>

        <div className="space-y-5 lg:col-span-3">
          {action.pending && <Loading label="Extracting patch embeddings and scoring" />}
          {Boolean(action.error) && <ErrorBanner error={action.error} onRetry={submit} />}

          {!action.pending && !result && (
            <Panel title="Awaiting an image">
              <Notice tone="info" title="No inspection yet">
                <p>
                  Upload a product image. The detector compares local patch descriptors against a
                  coreset memory bank built from normal MVTec images of the same category.
                </p>
                <p className="mt-2">
                  PatchCore is category-specific: a bank trained on <code>bottle</code> cannot judge
                  a <code>screw</code>, because &ldquo;normal&rdquo; is defined relative to one
                  product geometry.
                </p>
              </Notice>
            </Panel>
          )}

          {result && (
            <>
              {!result.model_trained && (
                <Notice tone="warn" title="No inspection was performed">
                  <p>{result.message}</p>
                </Notice>
              )}

              <Panel
                title="Verdict"
                subtitle={result.model_trained ? `${result.model_version} · ${result.category}` : undefined}
                actions={result.model_trained ? <Badge value={result.verdict} /> : undefined}
              >
                <div className="grid gap-4 sm:grid-cols-4">
                  <StatCard label="Anomalous" value={String(result.is_anomalous ?? '—')} tone={result.is_anomalous ? 'bad' : 'good'} />
                  <StatCard label="Score" value={result.anomaly_score?.toFixed(3) ?? '—'} />
                  <StatCard label="Threshold" value={result.threshold?.toFixed(3) ?? '—'} />
                  <StatCard label="Localisation" value={result.localisation_pct !== null ? `${result.localisation_pct?.toFixed(2)}%` : '—'} unit="of part" />
                </div>
                <p className="mt-4 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-[12px] leading-relaxed text-slate-300">
                  {result.message}
                </p>
              </Panel>

              {(result.original_url || result.heatmap_url || result.overlay_url) && (
                <Panel title="Original vs analysed" subtitle="Uploaded part, anomaly heatmap, annotated overlay">
                  <div className="grid gap-4 sm:grid-cols-3">
                    <Figure title="Original" src={result.original_url} />
                    <Figure title="Anomaly heatmap" src={result.heatmap_url} />
                    <Figure title="Annotated overlay" src={result.overlay_url} />
                  </div>
                  {result.overlay_url && (
                    <a
                      className="btn-secondary mt-4"
                      href={storageUrl(result.overlay_url) ?? '#'}
                      download
                    >
                      <Download className="h-4 w-4" /> Download overlay
                    </a>
                  )}
                </Panel>
              )}

              {result.preprocessing && (
                <Panel title="Preprocessing applied" subtitle="Exactly what happened to the pixels">
                  <pre className="mono overflow-x-auto rounded-lg border border-white/[0.06] bg-navy-950/60 p-3 text-[11px] text-slate-400">
                    {JSON.stringify(result.preprocessing, null, 2)}
                  </pre>
                </Panel>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function Figure({ title, src }: { title: string; src: string | null }) {
  const url = storageUrl(src)
  return (
    <figure>
      <div className="grid aspect-square place-items-center overflow-hidden rounded-lg border border-white/[0.06] bg-navy-950/60">
        {url ? (
          <img src={url} alt={title} className="h-full w-full object-contain" />
        ) : (
          <span className="text-[11px] text-slate-600">not available</span>
        )}
      </div>
      <figcaption className="mt-2 text-center text-[11px] text-slate-500">{title}</figcaption>
    </figure>
  )
}
