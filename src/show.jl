function Base.show(io::IO, ::MIME"text/plain", d::SynthesisDataset{T}) where T
    vals = [x.value for x in d.value]
    println(io, "SynthesisDataset{$T}")
    println(io, "  name  : $(d.name)")
    println(io, "  time  : $(length(d.time)) points in [$(minimum(d.time)), $(maximum(d.time))] Ma")
    print(  io, "  value : $(length(d.value)) $T in [$(round(minimum(vals), digits=3)), $(round(maximum(vals), digits=3))]")
end

function Base.show(io::IO, ::MIME"text/plain", d::RawData{T}) where T
    vals = [x.value for x in d.value]
    println(io, "RawData{$T}")
    println(io, "  name         : $(d.name)")
    println(io, "  location     : ($(d.lat)°, $(d.lon)°)")
    println(io, "  sample_id    : $(d.sample_id)")
    println(io, "  scale_factor : $(d.scale_factor)")
    println(io, "  rpi_mean     : $(d.rpi_mean)")
    println(io, "  time         : $(length(d.time)) points in [$(minimum(d.time)), $(maximum(d.time))] Ma")
    print(  io, "  value        : $(length(d.value)) $T in [$(round(minimum(vals), digits=3)), $(round(maximum(vals), digits=3))]")
end

function Base.show(io::IO, ::MIME"text/plain", d::PolarityDataset)
    println(io, "PolarityDataset")
    println(io, "  name     : $(d.name)")
    println(io, "  polarity : $(d.polarity)")
    println(io, "  chrons   : $(length(d.start)) intervals")
    print(  io, "  span     : [$(minimum(d.start)), $(maximum(d.stop))] Ma")
end
