import { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, Download } from 'lucide-react';

export default function AudioPlayer({ src, waveformData = [] }) {
    const [playing, setPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [duration, setDuration] = useState(0);
    const [muted, setMuted] = useState(false);
    const audioRef = useRef(null);
    const playheadRef = useRef(null);

    // Generate mock waveform if no data provided
    const peaks = waveformData.length > 0 ? waveformData : Array.from({ length: 60 }, () => Math.random() * 0.8 + 0.2);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const onTimeUpdate = () => setProgress((audio.currentTime / audio.duration) || 0);
        const onLoaded = () => setDuration(audio.duration);
        const onEnded = () => { setPlaying(false); setProgress(0); };

        audio.addEventListener('timeupdate', onTimeUpdate);
        audio.addEventListener('loadedmetadata', onLoaded);
        audio.addEventListener('ended', onEnded);

        return () => {
            audio.removeEventListener('timeupdate', onTimeUpdate);
            audio.removeEventListener('loadedmetadata', onLoaded);
            audio.removeEventListener('ended', onEnded);
        };
    }, []);

    function togglePlay() {
        if (!audioRef.current) return;
        if (playing) audioRef.current.pause();
        else audioRef.current.play();
        setPlaying(!playing);
    }

    function handleSeek(e) {
        if (!audioRef.current || !playheadRef.current) return;
        const rect = playheadRef.current.getBoundingClientRect();
        const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        audioRef.current.currentTime = percent * audioRef.current.duration;
        setProgress(percent);
    }

    const formatTime = (secs) => {
        if (!secs || isNaN(secs)) return '0:00';
        const m = Math.floor(secs / 60);
        const s = Math.floor(secs % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    return (
        <div className="bg-gray-900 rounded-2xl p-4 flex flex-col gap-3 w-full shadow-lg border border-gray-800">
            <audio ref={audioRef} src={src} preload="metadata" muted={muted} />

            {/* Top bar */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <button
                        onClick={togglePlay}
                        className="w-10 h-10 rounded-full bg-orange-500 hover:bg-orange-400 flex items-center justify-center text-white transition-colors"
                    >
                        {playing ? <Pause size={18} /> : <Play size={18} className="translate-x-0.5" />}
                    </button>
                    <div className="flex flex-col">
                        <span className="text-white font-semibold text-sm">Call Recording</span>
                        <span className="text-gray-400 text-xs font-mono">
                            {formatTime(audioRef.current?.currentTime || 0)} / {formatTime(duration)}
                        </span>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <button onClick={() => setMuted(!muted)} className="p-2 text-gray-400 hover:text-white transition-colors rounded-lg hover:bg-white/10">
                        {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
                    </button>
                    <a href={src} download className="p-2 text-gray-400 hover:text-white transition-colors rounded-lg hover:bg-white/10" title="Download Recording">
                        <Download size={16} />
                    </a>
                </div>
            </div>

            {/* Waveform Scrubber */}
            <div
                ref={playheadRef}
                className="h-10 w-full relative cursor-pointer group flex items-end gap-0.5"
                onClick={handleSeek}
            >
                {/* Background track (unplayed) */}
                {peaks.map((p, i) => {
                    const played = (i / peaks.length) <= progress;
                    return (
                        <div
                            key={i}
                            className={`flex-1 rounded-sm transition-colors duration-100 ${played ? 'bg-orange-500' : 'bg-gray-700 group-hover:bg-gray-600'}`}
                            style={{ height: `${p * 100}%`, minHeight: '4px' }}
                        />
                    );
                })}

                {/* Playhead line indicator */}
                <div
                    className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] pointer-events-none rounded-full"
                    style={{ left: `${progress * 100}%`, transition: playing ? 'none' : 'left 0.1s' }}
                />
            </div>
        </div>
    );
}
