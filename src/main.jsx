import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { FaceDetector, FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision'
import { removeBackground } from '@imgly/background-removal'
import JSZip from 'jszip'
import { Upload, Download, Trash2, Sparkles, SlidersHorizontal, Check, ImagePlus, ShieldCheck, RotateCcw, Package, X, Plus, Settings2, AlertTriangle, Eraser, Undo2 } from 'lucide-react'
import { templates, svgUrl } from './templates'
import './styles.css'

const OUTPUT_W = 600
const OUTPUT_H = 800
const MODEL_BASE = import.meta.env.BASE_URL
let faceDetectorPromise
let headSegmenterPromise
let visionFilesetPromise

function getVisionFileset() {
  if (!visionFilesetPromise) {
    // The JS package and WASM runtime must be the exact same version.
    visionFilesetPromise = FilesetResolver.forVisionTasks(`${MODEL_BASE}mediapipe/wasm`)
  }
  return visionFilesetPromise
}

function openTemplateDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('tongyan-studio', 1)
    request.onupgradeneeded = () => request.result.createObjectStore('templates', { keyPath: 'id' })
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

async function loadSavedTemplates() {
  const db = await openTemplateDb()
  return new Promise((resolve, reject) => {
    const request = db.transaction('templates', 'readonly').objectStore('templates').getAll()
    request.onsuccess = () => resolve(request.result || [])
    request.onerror = () => reject(request.error)
  })
}

async function saveCustomTemplate(template) {
  const db = await openTemplateDb()
  const stored = { ...template }
  delete stored.sourceSize
  return new Promise((resolve, reject) => {
    const request = db.transaction('templates', 'readwrite').objectStore('templates').put(stored)
    request.onsuccess = resolve
    request.onerror = () => reject(request.error)
  })
}

async function deleteCustomTemplate(id) {
  const db = await openTemplateDb()
  return new Promise((resolve, reject) => {
    const request = db.transaction('templates', 'readwrite').objectStore('templates').delete(id)
    request.onsuccess = resolve
    request.onerror = () => reject(request.error)
  })
}

function fileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

async function getFaceDetector() {
  if (!faceDetectorPromise) {
    faceDetectorPromise = getVisionFileset()
      .then(vision => FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `${MODEL_BASE}models/face_detector.tflite` },
        runningMode: 'IMAGE', minDetectionConfidence: 0.45
      }))
  }
  return faceDetectorPromise
}

async function getHeadSegmenter() {
  if (!headSegmenterPromise) {
    headSegmenterPromise = getVisionFileset()
      .then(vision => ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `${MODEL_BASE}models/selfie_multiclass.tflite` },
        runningMode: 'IMAGE', outputCategoryMask: true, outputConfidenceMasks: false
      }))
  }
  return headSegmenterPromise
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = reject
    image.src = src
  })
}

function templateUrl(template) {
  return template.imageUrl || svgUrl(template.svg)
}

function safeName(value) {
  return (value || '未命名').replace(/[\\/:*?"<>|]/g, '_').trim() || '未命名'
}

async function createHeadCutout(original, face, precise = false) {
  const fw = face.w, fh = face.h
  const sx = Math.max(0, face.x - fw * 0.72)
  const sy = Math.max(0, face.y - fh * 0.78)
  const sw = Math.min(original.naturalWidth - sx, fw * 2.44)
  // Stop shortly below the chin. Shoulders and torso never enter the model.
  const sh = Math.min(original.naturalHeight - sy, fh * 1.95)
  const scale = Math.min(1, (precise ? 512 : 384) / Math.max(sw, sh))
  const width = Math.max(1, Math.round(sw * scale))
  const height = Math.max(1, Math.round(sh * scale))
  const canvas = document.createElement('canvas')
  canvas.width = width; canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  ctx.drawImage(original, sx, sy, sw, sh, 0, 0, width, height)
  const headSegmenter = await getHeadSegmenter()
  const semanticResult = headSegmenter.segment(canvas)
  const categoryMask = semanticResult.categoryMask
  if (!categoryMask) throw new Error('The semantic head mask is unavailable.')
  const categories = categoryMask.getAsUint8Array()
  const semanticMask = document.createElement('canvas')
  semanticMask.width = categoryMask.width; semanticMask.height = categoryMask.height
  const semanticCtx = semanticMask.getContext('2d')
  const semanticPixels = semanticCtx.createImageData(semanticMask.width, semanticMask.height)
  for (let i = 0; i < categories.length; i++) {
    // Only head semantics survive: hair (1), face skin (3), accessories (5).
    // Body skin (2) removes hands/arms; clothes (4) and background (0) are rejected.
    const keep = categories[i] === 1 || categories[i] === 3 || categories[i] === 5
    const p = i * 4
    semanticPixels.data[p] = 255; semanticPixels.data[p + 1] = 255; semanticPixels.data[p + 2] = 255
    semanticPixels.data[p + 3] = keep ? 255 : 0
  }
  semanticCtx.putImageData(semanticPixels, 0, 0)
  categoryMask.close()

  const outputCanvas = document.createElement('canvas')
  outputCanvas.width = width
  outputCanvas.height = height
  const outputCtx = outputCanvas.getContext('2d')
  if (precise) {
    const inputBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
    const backgroundRemovedBlob = await removeBackground(inputBlob, {
      model: 'isnet_fp16', proxyToWorker: true, rescale: true,
      output: { format: 'image/png', quality: 1 }
    })
    const backgroundRemovedUrl = URL.createObjectURL(backgroundRemovedBlob)
    const backgroundRemoved = await loadImage(backgroundRemovedUrl)
    outputCtx.drawImage(backgroundRemoved, 0, 0, width, height)
    URL.revokeObjectURL(backgroundRemovedUrl)
  } else {
    outputCtx.drawImage(canvas, 0, 0)
  }
  outputCtx.globalCompositeOperation = 'destination-in'
  outputCtx.imageSmoothingEnabled = true
  outputCtx.filter = 'blur(0.8px)'
  outputCtx.drawImage(semanticMask, 0, 0, outputCanvas.width, outputCanvas.height)
  outputCtx.filter = 'none'
  outputCtx.globalCompositeOperation = 'source-over'
  const blob = await new Promise(resolve => outputCanvas.toBlob(resolve, 'image/png'))
  return {
    blob,
    crop: { x: 0, y: 0, w: outputCanvas.width, h: outputCanvas.height },
    face: {
      x: (face.x - sx) * scale * outputCanvas.width / width,
      y: (face.y - sy) * scale * outputCanvas.height / height,
      w: face.w * scale * outputCanvas.width / width,
      h: face.h * scale * outputCanvas.height / height
    }
  }
}

function createDetectionCanvas(image) {
  const maxSide = Math.max(image.naturalWidth, image.naturalHeight)
  if (maxSide <= 1280) return { source: image, scale: 1 }
  const scale = 1280 / maxSide
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(image.naturalWidth * scale)
  canvas.height = Math.round(image.naturalHeight * scale)
  canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height)
  return { source: canvas, scale }
}

function getPlacement(student, template) {
  const { crop, face } = student
  // Align the detected chin just above the collar. The remaining neck naturally
  // continues behind the body layer, so the result needs little manual tuning.
  const targetFaceWidth = template.head.w * 0.64
  const fit = (targetFaceWidth / Math.max(face.w, 1)) * student.scale
  const faceCenterInCrop = face.x - crop.x + face.w / 2
  const chinInCrop = face.y - crop.y + face.h
  // The generated template sheets place their neck openings slightly left and
  // above the geometric cell center. Apply a shared optical correction only
  // to built-in templates; user-calibrated custom templates keep exact anchors.
  const opticalX = template.custom ? 0 : -14
  const opticalY = template.custom ? 0 : -20
  const targetChinY = template.head.y + template.head.h * 0.45 + opticalY
  return {
    dw: crop.w * fit,
    dh: crop.h * fit,
    dx: template.head.x + opticalX - faceCenterInCrop * fit + student.offsetX,
    dy: targetChinY - chinInCrop * fit + student.offsetY
  }
}

function createErasedCutout(student) {
  if (!student.erasures?.length) return student.image
  const canvas = document.createElement('canvas')
  canvas.width = student.image.naturalWidth
  canvas.height = student.image.naturalHeight
  const ctx = canvas.getContext('2d')
  ctx.drawImage(student.image, 0, 0)
  ctx.globalCompositeOperation = 'destination-out'
  for (const point of student.erasures) {
    ctx.beginPath()
    ctx.arc(point.x, point.y, point.r, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalCompositeOperation = 'source-over'
  return canvas
}

async function prepareStudent(file, index, precise = false) {
  const startedAt = performance.now()
  const originalUrl = URL.createObjectURL(file)
  const original = await loadImage(originalUrl)
  let face = null
  try {
    const detector = await getFaceDetector()
    const detection = createDetectionCanvas(original)
    const result = detector.detect(detection.source)
    const box = result.detections?.[0]?.boundingBox || null
    face = box && {
      originX: box.originX / detection.scale, originY: box.originY / detection.scale,
      width: box.width / detection.scale, height: box.height / detection.scale
    }
  } catch (error) {
    console.warn('Face detection unavailable, using center crop.', error)
  }

  const fw = face?.width || original.naturalWidth * 0.4
  const fh = face?.height || original.naturalHeight * 0.4
  const fx = face?.originX ?? original.naturalWidth * 0.3
  const fy = face?.originY ?? original.naturalHeight * 0.16

  let cutoutUrl
  let processedCrop
  let processedFace
  try {
    const result = await createHeadCutout(original, { x: fx, y: fy, w: fw, h: fh }, precise)
    cutoutUrl = URL.createObjectURL(result.blob)
    processedCrop = result.crop
    processedFace = result.face
  } catch (error) {
    console.error('Head segmentation failed.', error)
    URL.revokeObjectURL(originalUrl)
    throw new Error('头部分割失败，请检查网络后重试')
  }

  const cutout = await loadImage(cutoutUrl)
  if (processedCrop && (cutout.naturalWidth !== processedCrop.w || cutout.naturalHeight !== processedCrop.h)) {
    const scaleX = cutout.naturalWidth / processedCrop.w
    const scaleY = cutout.naturalHeight / processedCrop.h
    processedFace = {
      x: processedFace.x * scaleX, y: processedFace.y * scaleY,
      w: processedFace.w * scaleX, h: processedFace.h * scaleY
    }
    processedCrop = { x: 0, y: 0, w: cutout.naturalWidth, h: cutout.naturalHeight }
  }
  const crop = processedCrop

  return {
    id: crypto.randomUUID(), name: file.name.replace(/\.[^.]+$/, '') || `小朋友${index + 1}`,
    file, originalUrl, cutoutUrl, image: cutout, crop,
    face: processedFace, templateId: templates[0].id,
    scale: 1, offsetX: 0, offsetY: 0, erasures: [], status: 'ready',
    cutoutMode: precise ? 'precise' : 'fast', durationMs: Math.round(performance.now() - startedAt)
  }
}

async function runWithConcurrency(items, limit, worker) {
  let cursor = 0
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++
      await worker(items[index], index)
    }
  })
  await Promise.all(runners)
}

function PreviewCanvas({ student, template, className = '', eraseMode = false, brushSize = 36, onErase }) {
  const canvasRef = useRef(null)
  const placementRef = useRef(null)
  const drawingRef = useRef(false)
  useEffect(() => {
    if (!student || !template) return
    let cancelled = false
    loadImage(templateUrl(template)).then(body => {
      if (cancelled) return
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, OUTPUT_W, OUTPUT_H)
      const { crop } = student
      const { dw, dh, dx, dy } = getPlacement(student, template)
      placementRef.current = { dw, dh, dx, dy }
      // Draw the body first. The extracted head sits above the collar so the
      // template cannot cover the child's mouth or chin.
      ctx.drawImage(body, 0, 0, OUTPUT_W, OUTPUT_H)
      ctx.drawImage(createErasedCutout(student), crop.x, crop.y, crop.w, crop.h, dx, dy, dw, dh)
    })
    return () => { cancelled = true }
  }, [student, template])
  const eraseAtPointer = event => {
    if (!eraseMode || !placementRef.current || !student) return
    const canvas = canvasRef.current
    const bounds = canvas.getBoundingClientRect()
    const canvasX = (event.clientX - bounds.left) * canvas.width / bounds.width
    const canvasY = (event.clientY - bounds.top) * canvas.height / bounds.height
    const { dw, dh, dx, dy } = placementRef.current
    const sourceX = student.crop.x + (canvasX - dx) * student.crop.w / dw
    const sourceY = student.crop.y + (canvasY - dy) * student.crop.h / dh
    if (sourceX < 0 || sourceY < 0 || sourceX > student.image.naturalWidth || sourceY > student.image.naturalHeight) return
    onErase?.({ x: sourceX, y: sourceY, r: brushSize * student.crop.w / dw / 2 })
  }
  return <canvas ref={canvasRef} width={OUTPUT_W} height={OUTPUT_H}
    className={`${className} ${eraseMode ? 'erase-active' : ''}`}
    aria-label={`${student?.name || ''}的合成预览`}
    onPointerDown={event => { if (!eraseMode) return; drawingRef.current = true; event.currentTarget.setPointerCapture(event.pointerId); eraseAtPointer(event) }}
    onPointerMove={event => { if (drawingRef.current) eraseAtPointer(event) }}
    onPointerUp={() => { drawingRef.current = false }} onPointerCancel={() => { drawingRef.current = false }}/>
}

function App() {
  const [students, setStudents] = useState([])
  const [customTemplates, setCustomTemplates] = useState([])
  const [templateCategory, setTemplateCategory] = useState('过年')
  const [uploadCategory, setUploadCategory] = useState('新年')
  const [selectedId, setSelectedId] = useState(null)
  const [processing, setProcessing] = useState({ active: false, done: 0, total: 0, label: '' })
  const [errorMessage, setErrorMessage] = useState('')
  const [dragging, setDragging] = useState(false)
  const [eraseMode, setEraseMode] = useState(false)
  const [refiningId, setRefiningId] = useState(null)
  const [brushSize, setBrushSize] = useState(38)
  const fileInput = useRef(null)
  const templateInput = useRef(null)
  const allTemplates = useMemo(() => [...templates, ...customTemplates], [customTemplates])
  const categories = useMemo(() => ['全部', ...new Set(allTemplates.map(t => t.category || '其他'))], [allTemplates])
  const visibleTemplates = templateCategory === '全部' ? allTemplates : allTemplates.filter(t => (t.category || '其他') === templateCategory)
  const selected = students.find(s => s.id === selectedId) || students[0]
  const selectedTemplate = allTemplates.find(t => t.id === selected?.templateId) || allTemplates[0]

  useEffect(() => { if (!selectedId && students[0]) setSelectedId(students[0].id) }, [students, selectedId])
  useEffect(() => {
    loadSavedTemplates().then(setCustomTemplates).catch(error => console.warn('Unable to load saved templates.', error))
  }, [])
  useEffect(() => {
    const warmUp = () => Promise.allSettled([getFaceDetector(), getHeadSegmenter()])
    const idleId = 'requestIdleCallback' in window
      ? window.requestIdleCallback(warmUp, { timeout: 1500 })
      : window.setTimeout(warmUp, 300)
    return () => 'cancelIdleCallback' in window ? window.cancelIdleCallback(idleId) : clearTimeout(idleId)
  }, [])

  const addFiles = useCallback(async fileList => {
    const files = [...fileList].filter(file => file.type.startsWith('image/'))
    if (!files.length) return
    setProcessing({ active: true, done: 0, total: files.length, label: '正在识别人像并抠图' })
    const added = new Array(files.length)
    const failed = []
    let finished = 0
    await runWithConcurrency(files, 2, async (file, i) => {
      try { added[i] = await prepareStudent(file, students.length + i) } catch (error) { failed.push(file.name); console.error(error) }
      finished += 1
      setProcessing(p => ({ ...p, done: finished }))
    })
    const completed = added.filter(Boolean)
    setStudents(current => [...current, ...completed])
    if (completed[0]) setSelectedId(completed[0].id)
    setProcessing({ active: false, done: 0, total: 0, label: '' })
    if (failed.length) setErrorMessage(`${failed.length} 张照片未能完成头部分割，请检查网络后重新上传。`)
  }, [students.length])

  const refineSelected = async () => {
    if (!selected || refiningId) return
    setRefiningId(selected.id)
    setProcessing({ active: true, done: 0, total: 1, label: '正在精细抠图' })
    try {
      const refined = await prepareStudent(selected.file, 0, true)
      URL.revokeObjectURL(refined.originalUrl)
      URL.revokeObjectURL(selected.cutoutUrl)
      setStudents(list => list.map(student => student.id === selected.id ? {
        ...student, cutoutUrl: refined.cutoutUrl, image: refined.image, crop: refined.crop,
        face: refined.face, cutoutMode: 'precise', durationMs: refined.durationMs, erasures: []
      } : student))
    } catch (error) {
      console.error(error)
      setErrorMessage('精细抠图失败，已保留当前快速抠图结果。')
    } finally {
      setRefiningId(null)
      setProcessing({ active: false, done: 0, total: 0, label: '' })
    }
  }

  const updateSelected = patch => setStudents(list => list.map(s => s.id === selected?.id ? { ...s, ...patch } : s))
  const addSelectedErasure = point => setStudents(list => list.map(s => s.id === selected?.id
    ? { ...s, erasures: [...(s.erasures || []), point] }
    : s))
  const removeStudent = id => {
    setStudents(list => list.filter(s => s.id !== id))
    if (selectedId === id) setSelectedId(null)
  }
  const applyTemplateToAll = templateId => setStudents(list => list.map(s => ({ ...s, templateId })))
  const addCustomTemplates = async fileList => {
    const files = [...(fileList || [])].filter(file => file.type.startsWith('image/'))
    if (!files.length) return
    const added = []
    for (const file of files) {
      const imageUrl = await fileAsDataUrl(file)
      const image = await loadImage(imageUrl)
      const next = {
        id: `custom-${crypto.randomUUID()}`, name: file.name.replace(/\.[^.]+$/, '') || '我的模板',
        category: uploadCategory.trim() || '其他', tone: '#89b7a8', imageUrl, custom: true,
        head: { x: 300, y: 145, w: 225, h: 235 }, sourceSize: { w: image.naturalWidth, h: image.naturalHeight }
      }
      added.push(next)
      await saveCustomTemplate(next)
    }
    setCustomTemplates(list => [...list, ...added])
    setTemplateCategory(uploadCategory.trim() || '其他')
    if (selected && added[0]) updateSelected({ templateId: added[0].id })
    templateInput.current.value = ''
  }
  const updateCustomTemplate = (id, headPatch) => setCustomTemplates(list => list.map(t => {
    if (t.id !== id) return t
    const updated = { ...t, head: { ...t.head, ...headPatch } }
    saveCustomTemplate(updated).catch(error => console.warn('Unable to save template calibration.', error))
    return updated
  }))
  const removeCustomTemplate = id => {
    setCustomTemplates(list => list.filter(t => t.id !== id))
    setStudents(list => list.map(s => s.templateId === id ? { ...s, templateId: templates[0].id } : s))
    deleteCustomTemplate(id).catch(error => console.warn('Unable to delete saved template.', error))
  }

  const renderBlob = async student => {
    const template = allTemplates.find(t => t.id === student.templateId) || allTemplates[0]
    const body = await loadImage(templateUrl(template))
    const canvas = document.createElement('canvas')
    canvas.width = OUTPUT_W; canvas.height = OUTPUT_H
    const ctx = canvas.getContext('2d')
    const { crop } = student
    const { dw, dh, dx, dy } = getPlacement(student, template)
    ctx.drawImage(body, 0, 0, OUTPUT_W, OUTPUT_H)
    ctx.drawImage(createErasedCutout(student), crop.x, crop.y, crop.w, crop.h, dx, dy, dw, dh)
    return new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
  }

  const exportOne = async student => {
    const blob = await renderBlob(student)
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = `${safeName(student.name)}_${allTemplates.find(t => t.id === student.templateId)?.name}.png`; a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }

  const exportAll = async () => {
    if (!students.length) return
    setProcessing({ active: true, done: 0, total: students.length, label: '正在打包透明 PNG' })
    const zip = new JSZip()
    for (let i = 0; i < students.length; i++) {
      const s = students[i], blob = await renderBlob(s)
      zip.file(`${String(i + 1).padStart(2, '0')}_${safeName(s.name)}.png`, blob)
      setProcessing(p => ({ ...p, done: i + 1 }))
    }
    const output = await zip.generateAsync({ type: 'blob' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(output); a.download = `卡通工坊_${students.length}人.zip`; a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
    setProcessing({ active: false, done: 0, total: 0, label: '' })
  }

  const empty = students.length === 0
  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Sparkles size={20}/></div><div><strong>卡通工坊</strong><span>班级卡通头像制作</span></div></div>
      <div className="privacy"><ShieldCheck size={16}/><span>照片仅在本机处理，不会上传</span></div>
      <button className="primary" onClick={exportAll} disabled={empty || processing.active}><Package size={17}/>批量导出 {students.length ? `${students.length} 张` : ''}</button>
    </header>

    <main className={empty ? 'workspace empty-workspace' : 'workspace'}>
      <aside className="student-panel">
        <div className="panel-heading"><div><span className="step">01</span><h2>学生照片</h2></div><span className="count">{students.length}</span></div>
        <button className="upload-small" onClick={() => fileInput.current?.click()}><ImagePlus size={17}/>添加照片</button>
        <input ref={fileInput} type="file" accept="image/*" multiple hidden onChange={e => addFiles(e.target.files)}/>
        <div className="student-list">
          {students.map((s, i) => <button key={s.id} className={`student-row ${s.id === selected?.id ? 'active' : ''}`} onClick={() => setSelectedId(s.id)}>
            <img src={s.originalUrl} alt=""/><span className="student-index">{String(i + 1).padStart(2, '0')}</span><span className="student-name">{s.name}</span>
            <span className="done"><Check size={13}/></span>
          </button>)}
        </div>
        {!empty && <button className="clear-link" onClick={() => { setStudents([]); setSelectedId(null) }}><Trash2 size={15}/>清空列表</button>}
      </aside>

      <section className="stage">
        {empty ? <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => {e.preventDefault(); setDragging(true)}} onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files)}}>
          <div className="upload-orbit"><Upload size={32}/></div>
          <p className="eyebrow">从这里开始</p><h1>上传学生照片<br/>一键变成卡通小人</h1>
          <p className="lead">建议使用证件照</p>
          <button className="primary large" onClick={() => fileInput.current?.click()}><Upload size={18}/>选择照片</button>
          <span className="or">或将多张照片拖到这里</span>
          <div className="flow"><span><b>1</b>自动抠图</span><i></i><span><b>2</b>选择模板</span><i></i><span><b>3</b>批量导出</span></div>
        </div> : <>
          <div className="stage-toolbar">
            <div><span className="step">02</span><h2>调整效果</h2></div>
            <div className="toolbar-actions"><button className="secondary refine-btn" title="复杂背景使用更精细但较慢的抠图" onClick={refineSelected} disabled={selected.cutoutMode === 'precise' || !!refiningId}><Sparkles size={16}/>{selected.cutoutMode === 'precise' ? '已精细处理' : '精细抠图'}</button><button className={`secondary erase-toggle ${eraseMode ? 'active' : ''}`} title="消除残留背景" onClick={() => setEraseMode(value => !value)}><Eraser size={16}/>消除</button><button className="icon-btn" title="重置位置" onClick={() => updateSelected({scale: 1, offsetX: 0, offsetY: 0})}><RotateCcw size={17}/></button><button className="secondary" onClick={() => exportOne(selected)}><Download size={16}/>导出当前</button></div>
          </div>
          <div className="preview-wrap"><div className="checker"><PreviewCanvas student={selected} template={selectedTemplate} className="main-preview" eraseMode={eraseMode} brushSize={brushSize} onErase={addSelectedErasure}/></div></div>
          {eraseMode && <div className="erase-toolbar">
            <Eraser size={16}/><label><span>笔刷大小</span><input type="range" min="12" max="100" value={brushSize} onChange={e => setBrushSize(+e.target.value)}/></label>
            <span className="brush-dot" style={{width: Math.max(8, brushSize * .32), height: Math.max(8, brushSize * .32)}}/>
            <button title="撤销上一步" disabled={!selected.erasures?.length} onClick={() => updateSelected({erasures: selected.erasures.slice(0, -1)})}><Undo2 size={15}/>撤销</button>
            <button title="还原全部消除" disabled={!selected.erasures?.length} onClick={() => updateSelected({erasures: []})}><RotateCcw size={15}/>还原</button>
          </div>}
          <div className="process-summary"><span className={selected.cutoutMode === 'precise' ? 'precise' : ''}>{selected.cutoutMode === 'precise' ? '精细模式' : '快速模式'}</span><b>{selected.durationMs ? `${(selected.durationMs / 1000).toFixed(1)} 秒` : ''}</b></div>
          <div className="student-editor">
            <label><span>文件名</span><input value={selected.name} onChange={e => updateSelected({name: e.target.value})}/></label>
            <label><span>头像大小</span><input type="range" min="0.75" max="1.35" step="0.01" value={selected.scale} onChange={e => updateSelected({scale: +e.target.value})}/><output>{Math.round(selected.scale * 100)}%</output></label>
            <label><span>左右位置</span><input type="range" min="-70" max="70" value={selected.offsetX} onChange={e => updateSelected({offsetX: +e.target.value})}/></label>
            <label><span>上下位置</span><input type="range" min="-70" max="70" value={selected.offsetY} onChange={e => updateSelected({offsetY: +e.target.value})}/></label>
          </div>
        </>}
      </section>

      <aside className="template-panel">
        <div className="panel-heading"><div><span className="step">03</span><h2>选择模板</h2></div></div>
        <p className="panel-note">点击模板应用到当前学生</p>
        <div className="category-tabs" aria-label="模板分类">
          {categories.map(category => <button key={category} className={templateCategory === category ? 'active' : ''} onClick={() => setTemplateCategory(category)}>{category}</button>)}
        </div>
        <div className="template-grid">
          {visibleTemplates.map(t => <button key={t.id} className={`template-card ${selected?.templateId === t.id ? 'active' : ''}`} onClick={() => selected && updateSelected({templateId: t.id})} disabled={empty}>
            <div className="template-art" style={{'--tone': t.tone}}><img src={templateUrl(t)} alt=""/></div><span>{t.name}</span>{selected?.templateId === t.id && <b><Check size={12}/></b>}
          </button>)}
        </div>
        <div className="template-upload-row">
          <input aria-label="模板分类" value={uploadCategory} onChange={e => setUploadCategory(e.target.value)} placeholder="输入分类，如：新年"/>
          <button className="upload-template" onClick={() => templateInput.current?.click()}><Plus size={16}/>批量上传</button>
        </div>
        <input ref={templateInput} type="file" accept="image/png,image/webp,image/jpeg" multiple hidden onChange={e => addCustomTemplates(e.target.files)}/>
        <p className="template-hint">推荐 3:4 透明 PNG，并在头部位置留空</p>
        {selectedTemplate?.custom && <div className="template-calibration">
          <div className="calibration-title"><span><Settings2 size={14}/>校准头部位置</span><button title="删除自定义模板" onClick={() => removeCustomTemplate(selectedTemplate.id)}><Trash2 size={14}/></button></div>
          <label><span>左右</span><input type="range" min="80" max="520" value={selectedTemplate.head.x} onChange={e => updateCustomTemplate(selectedTemplate.id,{x:+e.target.value})}/></label>
          <label><span>上下</span><input type="range" min="60" max="360" value={selectedTemplate.head.y} onChange={e => updateCustomTemplate(selectedTemplate.id,{y:+e.target.value})}/></label>
          <label><span>宽度</span><input type="range" min="100" max="360" value={selectedTemplate.head.w} onChange={e => updateCustomTemplate(selectedTemplate.id,{w:+e.target.value})}/></label>
          <label><span>高度</span><input type="range" min="120" max="380" value={selectedTemplate.head.h} onChange={e => updateCustomTemplate(selectedTemplate.id,{h:+e.target.value})}/></label>
        </div>}
        {!empty && <button className="apply-all" onClick={() => applyTemplateToAll(selected.templateId)}><SlidersHorizontal size={16}/>应用到全部学生</button>}
        <div className="tip"><strong>拍照小提示</strong><p>正面看镜头、头发完整、背景简洁，自动抠图效果更好。</p></div>
      </aside>
    </main>

    {processing.active && <div className="modal-backdrop"><div className="progress-modal"><button className="modal-x" aria-label="关闭" disabled><X size={16}/></button><div className="spinner"><Sparkles size={22}/></div><h3>{processing.label}</h3><p>{processing.done} / {processing.total}</p><div className="progress-track"><i style={{width: `${processing.total ? processing.done / processing.total * 100 : 5}%`}}/></div><span>{processing.label.includes('精细') ? '复杂照片会使用高质量模型，请稍候' : '快速模式通常可在 20 秒内完成'}</span></div></div>}
    {errorMessage && <div className="error-toast" role="alert"><AlertTriangle size={18}/><span>{errorMessage}</span><button aria-label="关闭提示" onClick={() => setErrorMessage('')}><X size={16}/></button></div>}
  </div>
}

const appRoot = globalThis.__tongyanAppRoot || createRoot(document.getElementById('root'))
globalThis.__tongyanAppRoot = appRoot
appRoot.render(<App />)
