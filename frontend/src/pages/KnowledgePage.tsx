import { Database, FileText, MessageSquare, Trash2, Upload } from 'lucide-react'
import { useRef, useState } from 'react'

import { Badge, EmptyState, ErrorBanner, Loading, Notice, Panel, StatCard } from '@/components/ui'
import { useAction, useAsync } from '@/hooks/useApi'
import {
  deleteDocument,
  getDocuments,
  getIndexHealth,
  queryKnowledge,
  uploadDocument,
} from '@/lib/endpoints'
import type { KnowledgeQueryResponse, KnowledgeUploadResponse } from '@/types/api'

export default function KnowledgePage() {
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [question, setQuestion] = useState('')

  const health = useAsync(() => getIndexHealth(), [])
  const documents = useAsync(() => getDocuments(), [])
  const upload = useAction<[File, string | undefined], KnowledgeUploadResponse>(uploadDocument)
  const ask = useAction<[string, number?], KnowledgeQueryResponse>(queryKnowledge)
  const remove = useAction<[string], boolean>(deleteDocument)

  const submitUpload = () => {
    if (!file) return
    upload.run(file, title.trim() || undefined).then((result) => {
      if (result) {
        setFile(null)
        setTitle('')
        if (fileRef.current) fileRef.current.value = ''
        documents.refetch()
        health.refetch()
      }
    })
  }

  const submitQuestion = () => {
    const trimmed = question.trim()
    if (trimmed.length >= 3) ask.run(trimmed, 5)
  }

  const removeDocument = (id: string) => {
    remove.run(id).then((ok) => {
      if (ok) {
        documents.refetch()
        health.refetch()
      }
    })
  }

  const index = health.data

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">Knowledge Base</h1>
        <p className="mt-1 text-[13px] text-slate-400">
          Retrieval-augmented generation over indexed manuals. Every answer is assembled from
          retrieved passages and cites them; when nothing relevant is indexed the API returns
          insufficient evidence instead of an unsourced claim.
        </p>
      </div>

      {index && !index.embedding_available && (
        <Notice tone="warn" title="Semantic retrieval is disabled">
          <p>{index.detail ?? 'The embedding model could not be loaded, so queries cannot run.'}</p>
        </Notice>
      )}
      {Boolean(health.error) && <ErrorBanner error={health.error} onRetry={health.refetch} />}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Indexed chunks" value={index?.chunks ?? 0} icon={<Database className="h-4 w-4" />} />
        <StatCard label="Documents" value={index?.documents ?? 0} icon={<FileText className="h-4 w-4" />} />
        <StatCard
          label="Embedding model"
          value={index?.embedding_model ?? '—'}
          hint={index?.backend ? `${index.backend} backend` : undefined}
        />
        <StatCard
          label="Index status"
          value={<Badge value={index?.status ?? (index?.embedding_available ? 'ok' : 'blocked')} />}
          hint={index?.dimension ? `${index.dimension} dimensions` : undefined}
          tone={index?.embedding_available ? 'good' : 'bad'}
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Upload a document" subtitle="PDF, TXT, MD, RST, LOG or CSV">
          <div className="space-y-4">
            <div
              onClick={() => fileRef.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault()
                const dropped = event.dataTransfer.files?.[0]
                if (dropped) setFile(dropped)
              }}
              className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-white/10 px-6 py-10 text-center transition hover:border-cyan-brand/40 hover:bg-white/[0.02]"
            >
              <Upload className="h-6 w-6 text-slate-600" />
              <p className="text-[13px] text-slate-400">
                {file ? file.name : 'Drop a manual or click to browse'}
              </p>
              {file && (
                <p className="mono text-[11px] text-slate-600">{(file.size / 1024).toFixed(0)} KB</p>
              )}
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,.txt,.md,.rst,.log,.csv"
                className="hidden"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </div>

            <div>
              <label className="label" htmlFor="doc_title">
                Title (optional)
              </label>
              <input
                id="doc_title"
                type="text"
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                className="input"
                placeholder="Defaults to the file name"
              />
            </div>

            <button
              className="btn-primary w-full"
              disabled={!file || upload.pending}
              onClick={submitUpload}
            >
              <Upload className="h-4 w-4" />
              {upload.pending ? 'Extracting and embedding...' : 'Upload and index'}
            </button>

            {Boolean(upload.error) && <ErrorBanner error={upload.error} onRetry={submitUpload} />}

            {upload.result && (
              <Notice tone="good" title={`Indexed ${upload.result.chunk_count} chunks`}>
                <p>{upload.result.message}</p>
                <p className="mono mt-2 text-[11px] text-slate-500">
                  {upload.result.embedding_model} · {upload.result.char_count} chars · index size{' '}
                  {upload.result.index_size}
                </p>
              </Notice>
            )}
          </div>
        </Panel>

        <Panel title="Ask the knowledge base" subtitle="Answers are limited to indexed passages">
          <div className="space-y-4">
            <div>
              <label className="label" htmlFor="question">
                Question
              </label>
              <textarea
                id="question"
                rows={3}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) submitQuestion()
                }}
                className="input resize-none"
                placeholder="What torque limit triggers a tool-change inspection?"
              />
            </div>
            <button
              className="btn-primary w-full"
              disabled={ask.pending || question.trim().length < 3}
              onClick={submitQuestion}
            >
              <MessageSquare className="h-4 w-4" />
              {ask.pending ? 'Retrieving...' : 'Query index'}
            </button>

            {Boolean(ask.error) && <ErrorBanner error={ask.error} onRetry={submitQuestion} />}

            {ask.result && <Answer result={ask.result} />}
          </div>
        </Panel>
      </div>

      <Panel
        title="Indexed documents"
        subtitle={`${documents.data?.length ?? 0} registered`}
        actions={
          <button className="btn-ghost" onClick={documents.refetch}>
            Refresh
          </button>
        }
        dense
      >
        {documents.loading ? (
          <Loading label="Loading documents" />
        ) : !documents.data?.length ? (
          <EmptyState
            title="No documents indexed"
            detail="Upload a manual or specification so the documentation agent can cite it."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[12px]">
              <thead>
                <tr className="border-b border-white/[0.08]">
                  {['Document', 'Title', 'Chunks', 'Pages', 'Embedding', 'Status', 'Indexed', ''].map(
                    (header) => (
                      <th
                        key={header}
                        className="px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500"
                      >
                        {header}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {documents.data.map((doc) => (
                  <tr key={doc.id} className="border-b border-white/[0.04] last:border-0">
                    <td className="px-3 py-2 text-slate-200">{doc.filename}</td>
                    <td className="px-3 py-2 text-slate-400">{doc.title ?? '—'}</td>
                    <td className="mono px-3 py-2 text-slate-300">{doc.chunk_count ?? '—'}</td>
                    <td className="mono px-3 py-2 text-slate-300">{doc.page_count ?? '—'}</td>
                    <td className="mono px-3 py-2 text-[10px] text-slate-500">
                      {doc.embedding_model ?? '—'}
                    </td>
                    <td className="px-3 py-2">
                      <Badge value={doc.status} />
                    </td>
                    <td className="mono px-3 py-2 text-[10px] text-slate-500">
                      {new Date(doc.created_at).toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <button
                        className="btn-ghost px-2 py-1 text-[11px] text-risk-critical"
                        disabled={remove.pending}
                        onClick={() => removeDocument(doc.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {index?.index_path && (
        <p className="mono break-all text-[10px] text-slate-600">index: {index.index_path}</p>
      )}
    </div>
  )
}

function Answer({ result }: { result: KnowledgeQueryResponse }) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge
          value={result.insufficient_evidence ? 'insufficient_evidence' : 'ok'}
          label={result.answer_engine.replace(/_/g, ' ').toUpperCase()}
        />
        <span className="mono text-[10px] text-slate-600">
          {result.retrieved_count} passage{result.retrieved_count === 1 ? '' : 's'} retrieved
        </span>
      </div>

      <p className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-[12.5px] leading-relaxed text-slate-200">
        {result.answer}
      </p>

      {result.citations.length > 0 ? (
        <ul className="space-y-2">
          {result.citations.map((chunk) => (
            <li
              key={chunk.chunk_id}
              className="rounded-lg border border-white/[0.06] bg-navy-950/50 p-3"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="mono text-[10px] text-cyan-brand">
                  [{chunk.chunk_id}] {chunk.document_title ?? chunk.filename}
                  {chunk.page ? ` p.${chunk.page}` : ''}
                </span>
                <span className="mono text-[10px] text-slate-600">
                  score {chunk.score.toFixed(3)}
                </span>
              </div>
              <p className="mt-1.5 text-[11.5px] leading-relaxed text-slate-400">{chunk.text}</p>
            </li>
          ))}
        </ul>
      ) : (
        <Notice tone="warn" title="No citation resolved">
          <p>
            Nothing in the index scored above the retrieval threshold, so the API declined to answer
            rather than invent a source.
          </p>
        </Notice>
      )}

      <p className="text-[11px] leading-relaxed text-slate-600">{result.disclaimer}</p>
    </div>
  )
}
