

interface WindowConfigureEvent {
    id: "win:conf";
    size?: [x: number, y: number];
    scale?: number; // scale applied to size
}

interface WindowFocusEvent {
    id: "win:focus";
}


interface VideoConfigureEvent {
    id: "vid:conf";
    codec: string;
    description: ArrayBuffer;
}

interface VideoDecodeEvent {
    id: "vid:dec";
    type: "key" | "delta";
    timestamp: number; // TODO in us
    data: ArrayBuffer;
}


interface ClipboardInputEvent {
    id: "inp:clb";
    items: {[mime: string]: ArrayBuffer};
}

interface KeyboardInputEvent {
    id: "inp:kb";
    action?: "up" | "down" | null;
    key: string;
}

interface PointerInputEvent {
    id: "inp:pt";
    action?: "up" | "down" | null;
    button?: number | null;
    pos?: [x: number, y: number];
}

interface WheelInputEvent {
    id: "inp:whl";
    deltaPos: [x: number, y: number, z: number];
}


type ViewerEvent = (
    | WindowConfigureEvent
    | WindowFocusEvent
    | VideoConfigureEvent
    | VideoDecodeEvent
    | ClipboardInputEvent
    | KeyboardInputEvent
    | PointerInputEvent
    | WheelInputEvent
);


interface ViewerCommunication {
    // TODO ack?
    emit: (event: ViewerEvent) => void;
    // TODO async??
    on?: (listener: (event: ViewerEvent) => void) => void;
    off?: (listener: (event: ViewerEvent) => void) => void;
}




interface ViewerOptions {
    comm?: ViewerCommunication;
}


function useViewer(canvas: HTMLCanvasElement, options: ViewerOptions) {
    const comm = options?.comm;
    const abortController = new AbortController();

    // NOTE this will be automatically called the first time
    const resizeObserver = new ResizeObserver(() => {
        const scale = window.devicePixelRatio ?? 1;
        const width = Math.floor(canvas.clientWidth  * scale);
        const height = Math.floor(canvas.clientHeight * scale);

        // NOTE setting .width or .height clears canvas buffer so use sparingly
        // NOTE this also prevents the remote from echoing
        if (canvas.width === width && canvas.height === height) 
            return;

        canvas.width = width; 
        canvas.height = height;            

        comm?.emit({
            id: "win:conf",
            // TODO should std use scaled size or unscaled?
            size: [canvas.width, canvas.height],
            scale: window.devicePixelRatio ?? 1,
        });
    });
    resizeObserver.observe(canvas);
    abortController.signal.addEventListener(
        "abort", 
        () => resizeObserver.disconnect(), 
        { once: true },
    );

    if (true) {
        const bitmapContext = canvas.getContext("bitmaprenderer");
        const decoder = new VideoDecoder({
            output: async (frame) => {
                // TODO rm
                // console.log(frame);
                try {
                    bitmapContext?.transferFromImageBitmap(
                        await createImageBitmap(frame)
                    );
                } finally {
                    frame.close();
                }
            },
            // TODO
            error: (e) => console.error('[decoder error]', e),
        });

        // TODO abort signal
        const videoHandler = (event: ViewerEvent) => {
            switch (event.id) {
                case "vid:conf":
                    // TODO
                    decoder.configure({
                        codec: event.codec,
                        description: event.description ?? undefined,
                        // TODO
                        hardwareAcceleration: 'prefer-hardware',
                        optimizeForLatency: true,
                        // latencyMode: 'realtime',
                    });
                    break;
                case "vid:dec":
                    const chunk = new EncodedVideoChunk({
                        type: event.type,
                        timestamp: event.timestamp,
                        data: event.data,
                        // transfer: [event.data.buffer],
                    });
                    decoder.decode(chunk);
                    break;
            }
        };
        comm?.on?.(videoHandler);
        abortController.signal.addEventListener(
            "abort", 
            () => comm?.off?.(videoHandler), 
            { once: true },
        );
    }

    // TODO

    if (true) {
        canvas.addEventListener(
            "click", 
            (event) => {
                event.preventDefault();
                canvas.focus();
                comm?.emit({id: "win:focus"});
            }, 
            { signal: abortController.signal, passive: false },
        );            
    }

    // TODO

    if (true) {
        canvas.addEventListener(
            "pointermove", 
            (event) => {
                event.preventDefault();
                comm?.emit({
                    id: "inp:pt",
                    pos: [event.offsetX, event.offsetY],
                });
            }, 
            { signal: abortController.signal, passive: false },
        );

        canvas.addEventListener(
            "pointerdown", 
            (event) => {
                event.preventDefault();            
                comm?.emit({
                    id: "inp:pt",
                    action: "down",
                    button: event.button,
                });
            }, 
            { signal: abortController.signal, passive: false },
        );

        canvas.addEventListener(
            "pointerup", 
            (event) => {
                event.preventDefault();            
                comm?.emit({
                    id: "inp:pt",
                    action: "up",
                    button: event.button,
                });
            }, 
            { signal: abortController.signal, passive: false },
        );

        canvas.addEventListener(
            "contextmenu", 
            (event) => event.preventDefault(), 
            { signal: abortController.signal, passive: false },
        );            
    }

    if (true) {
        canvas.addEventListener(
            "wheel", 
            (event) => {
                event.preventDefault();
                comm?.emit({
                    id: "inp:whl",
                    deltaPos: [event.deltaX, event.deltaY, event.deltaZ],
                });
            }, 
            { signal: abortController.signal, passive: false },
        );
    }

    if (true) {
        canvas.addEventListener(
            "keydown", 
            (event) => {
                event.preventDefault();
                comm?.emit({
                    id: "inp:kb",
                    action: "down",
                    key: event.key,
                });
            }, 
            { signal: abortController.signal, passive: false },
        );

        canvas.addEventListener(
            "keyup", 
            (event) => {
                event.preventDefault();
                comm?.emit({
                    id: "inp:kb",
                    action: "up",
                    key: event.key,
                });
            }, 
            { signal: abortController.signal, passive: false },
        );
    }

    // TODO clipboard

    if (true) {
        canvas.addEventListener(
            "focus",
            (event) => {
                // TODO
            },
            { signal: abortController.signal, passive: false },
        );

        canvas.addEventListener(
            "blur",
            (event) => {
                // TODO
            },
            { signal: abortController.signal, passive: false },
        );
    }

    return {
        disconnect() {
            abortController.abort();
        }
    };
}
