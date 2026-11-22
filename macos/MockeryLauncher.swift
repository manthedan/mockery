import AppKit
import Foundation

final class MockeryAppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private var serverProcess: Process?
    private var logHandle: FileHandle?
    private var attachedExternalServer: Bool = false
    private var activePort: Int = 8765
    private var statusMenuItem: NSMenuItem?
    private let dataDir: URL
    private let logFile: URL

    override init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        self.dataDir = appSupport.appendingPathComponent("Mockery", isDirectory: true)
        let logsDir = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Logs/Mockery", isDirectory: true)
        try? FileManager.default.createDirectory(at: self.dataDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
        self.logFile = logsDir.appendingPathComponent("server.log")
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMenuBar()
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let port = self.activePort
            let alreadyHealthy = self.isServerHealthy(port: port)
            DispatchQueue.main.async {
                if alreadyHealthy {
                    self.attachedExternalServer = true
                    self.updateStatus("Running on port \(port) (external)")
                    self.openStudioInBrowser()
                    return
                }
                self.startServer(openBrowserWhenReady: true)
            }
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        openStudioInBrowser()
        return true
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopServer()
    }

    private func setupMenuBar() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            if let img = NSImage(systemSymbolName: "record.circle", accessibilityDescription: "Mockery") {
                img.isTemplate = true
                button.image = img
            } else {
                button.title = "Mockery"
            }
            button.toolTip = "Mockery — Local Mock Interview Studio"
        }

        let menu = NSMenu()
        let titleItem = NSMenuItem(title: "Mockery", action: nil, keyEquivalent: "")
        titleItem.isEnabled = false
        menu.addItem(titleItem)

        let st = NSMenuItem(title: "Starting…", action: nil, keyEquivalent: "")
        st.isEnabled = false
        self.statusMenuItem = st
        menu.addItem(st)

        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Open Mockery", action: #selector(handleOpenBrowser), keyEquivalent: "o"))
        menu.addItem(NSMenuItem(title: "Open Saved Interviews Folder", action: #selector(handleOpenDataDir), keyEquivalent: ""))
        menu.addItem(NSMenuItem(title: "View Server Log", action: #selector(handleOpenLog), keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Restart Server", action: #selector(handleRestart), keyEquivalent: "r"))
        menu.addItem(NSMenuItem(title: "Quit Mockery", action: #selector(handleQuit), keyEquivalent: "q"))

        for mItem in menu.items where mItem.action != nil {
            mItem.target = self
        }
        item.menu = menu
        self.statusItem = item
    }

    private func updateStatus(_ text: String) {
        DispatchQueue.main.async {
            self.statusMenuItem?.title = text
        }
    }

    private func findMockeryExecutable() -> (String, [String])? {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser.path
        let candidates = [
            "\(home)/.local/bin/mockery",
            "/opt/homebrew/bin/mockery",
            "/usr/local/bin/mockery",
        ]
        for path in candidates where fm.isExecutableFile(atPath: path) {
            return (path, [])
        }
        let uvCandidates = [
            "\(home)/.local/bin/uv",
            "\(home)/.cargo/bin/uv",
            "/opt/homebrew/bin/uv",
            "/usr/local/bin/uv",
        ]
        for uvPath in uvCandidates where fm.isExecutableFile(atPath: uvPath) {
            return (uvPath, ["tool", "run", "--from", "git+https://github.com/manthedan/mockery.git", "mockery"])
        }
        return nil
    }

    private func startServer(openBrowserWhenReady: Bool) {
        stopServer()
        if attachedExternalServer && isServerHealthy(port: activePort) {
            updateStatus("Running on port \(activePort) (external)")
            showErrorAlert("Mockery is already running on port \(activePort) in another process (such as Terminal). Stop that server first before restarting from the menu bar.")
            return
        }
        attachedExternalServer = false
        guard let (exe, baseArgs) = findMockeryExecutable() else {
            updateStatus("Mockery CLI not installed")
            showInstallAlert()
            return
        }

        activePort = findAvailablePort(startingAt: 8765)
        updateStatus("Starting on port \(activePort)…")

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: exe)
        proc.arguments = baseArgs + [
            "--host", "127.0.0.1",
            "--port", String(activePort),
            "--data-dir", dataDir.path,
            "--no-browser",
        ]

        var env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let extraPath = "\(home)/.local/bin:/opt/homebrew/bin:/usr/local/bin"
        env["PATH"] = extraPath + ":" + (env["PATH"] ?? "/usr/bin:/bin")
        env["MOCKERY_HOME"] = dataDir.path
        proc.environment = env

        if !FileManager.default.fileExists(atPath: logFile.path) {
            FileManager.default.createFile(atPath: logFile.path, contents: nil)
        }
        if let fh = try? FileHandle(forWritingTo: logFile) {
            fh.seekToEndOfFile()
            let banner = "\n--- Mockery server launch (\(ISO8601DateFormatter().string(from: Date()))) on port \(activePort) ---\n"
            if let data = banner.data(using: .utf8) {
                fh.write(data)
            }
            self.logHandle = fh
            proc.standardOutput = fh
            proc.standardError = fh
        }

        proc.terminationHandler = { [weak self] terminated in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if self.serverProcess === terminated {
                    try? self.logHandle?.close()
                    self.logHandle = nil
                    self.serverProcess = nil
                }
            }
            if terminated.terminationReason == .exit && terminated.terminationStatus == 0 {
                self.updateStatus("Stopped")
            } else {
                self.updateStatus("Server exited (\(terminated.terminationStatus))")
            }
        }

        do {
            try proc.run()
            self.serverProcess = proc
        } catch {
            updateStatus("Failed to launch")
            showErrorAlert("Could not launch Mockery server: \(error.localizedDescription)")
            return
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let port = self.activePort
            for _ in 0..<80 {
                if self.isServerHealthy(port: port) {
                    self.updateStatus("Running on port \(port)")
                    if openBrowserWhenReady {
                        DispatchQueue.main.async {
                            self.openStudioInBrowser()
                        }
                    }
                    return
                }
                if self.serverProcess?.isRunning == false {
                    return
                }
                Thread.sleep(forTimeInterval: 0.25)
            }
            self.updateStatus("Starting (check log if slow)…")
        }
    }

    private func stopServer() {
        defer {
            try? logHandle?.close()
            logHandle = nil
        }
        guard let proc = serverProcess else { return }
        if proc.isRunning {
            proc.terminate()
            let deadline = Date().addingTimeInterval(2.5)
            while proc.isRunning && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
            if proc.isRunning {
                kill(proc.processIdentifier, SIGKILL)
            }
        }
        serverProcess = nil
    }

    private func findAvailablePort(startingAt start: Int) -> Int {
        for port in start..<(start + 20) {
            if !isPortBusy(port: port) {
                return port
            }
        }
        return start
    }

    private func isPortBusy(port: Int) -> Bool {
        let sock = socket(AF_INET, SOCK_STREAM, 0)
        if sock < 0 { return false }
        defer { close(sock) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(UInt16(port).bigEndian)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let res = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return res == 0
    }

    private func isServerHealthy(port: Int) -> Bool {
        guard let url = URL(string: "http://127.0.0.1:\(port)/api/machines") else { return false }
        var req = URLRequest(url: url)
        req.timeoutInterval = 0.5
        let sem = DispatchSemaphore(value: 0)
        var ok = false
        URLSession.shared.dataTask(with: req) { _, resp, _ in
            if let http = resp as? HTTPURLResponse, http.statusCode == 200 {
                ok = true
            }
            sem.signal()
        }.resume()
        _ = sem.wait(timeout: .now() + 0.7)
        return ok
    }

    private func openStudioInBrowser() {
        guard let url = URL(string: "http://127.0.0.1:\(activePort)") else { return }
        // Prefer Google Chrome when installed (best tab-audio capture for Google Meet)
        let chromeBundle = URL(fileURLWithPath: "/Applications/Google Chrome.app")
        if FileManager.default.fileExists(atPath: chromeBundle.path) {
            let cfg = NSWorkspace.OpenConfiguration()
            NSWorkspace.shared.open([url], withApplicationAt: chromeBundle, configuration: cfg) { _, err in
                if err != nil {
                    NSWorkspace.shared.open(url)
                }
            }
            return
        }
        NSWorkspace.shared.open(url)
    }

    private func showInstallAlert() {
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.messageText = "Mockery needs to be installed first"
            alert.informativeText = "Run the one-line installer in Terminal:\n\ncurl -fsSL https://raw.githubusercontent.com/manthedan/mockery/main/install.sh | sh"
            alert.alertStyle = .informational
            alert.addButton(withTitle: "Copy Command")
            alert.addButton(withTitle: "Quit")
            let resp = alert.runModal()
            if resp == .alertFirstButtonReturn {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(
                    "curl -fsSL https://raw.githubusercontent.com/manthedan/mockery/main/install.sh | sh",
                    forType: .string
                )
            }
            NSApp.terminate(nil)
        }
    }

    private func showErrorAlert(_ message: String) {
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.messageText = "Mockery"
            alert.informativeText = message
            alert.alertStyle = .warning
            alert.addButton(withTitle: "OK")
            alert.runModal()
        }
    }

    @objc private func handleOpenBrowser() {
        openStudioInBrowser()
    }

    @objc private func handleOpenDataDir() {
        let sessions = dataDir.appendingPathComponent("sessions", isDirectory: true)
        try? FileManager.default.createDirectory(at: sessions, withIntermediateDirectories: true)
        NSWorkspace.shared.open(sessions)
    }

    @objc private func handleOpenLog() {
        NSWorkspace.shared.open(logFile)
    }

    @objc private func handleRestart() {
        startServer(openBrowserWhenReady: true)
    }

    @objc private func handleQuit() {
        stopServer()
        NSApp.terminate(nil)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = MockeryAppDelegate()
app.delegate = delegate
app.run()
