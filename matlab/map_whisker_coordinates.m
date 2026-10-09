function [mappedXY,sourceFrameID0]=map_whisker_coordinates(points,metadata,frameID0,side,direction,inputBase,outputBase)
% Map points with the EXACT matrices saved by preprocess.py.
% Example: xyMain1=map_whisker_coordinates([x y],csvFile,frame0,'R','to_original',0,1);
% Janelia/OpenCV coordinates are zero-based. MATLAB imshow plotting is one-based.
% This helper cannot reconstruct coordinates from legacy videos without metadata.
if nargin<5||isempty(direction),direction='to_original';end
if nargin<6,inputBase=0;end
if nargin<7,outputBase=1;end
side=validatestring(upper(side),{'L','R'});
direction=validatestring(direction,{'to_original','to_processed'});
validateattributes(points,{'numeric'},{'2d','ncols',2});
validateattributes(frameID0,{'numeric'},{'scalar','integer','nonnegative'});
assert(ismember(inputBase,[0 1])&&ismember(outputBase,[0 1]),'Pixel bases must be 0 or 1.');
if istable(metadata),T=metadata;else,T=readtable(metadata);end
row=find(T.processed_frame0==frameID0);
assert(numel(row)==1,'Processed frame %d missing or duplicated in metadata.',frameID0);
assert(T.pose_valid(row)==1,'Frame %d has no usable head pose; do not infer a curve.',frameID0);
prefix=side;if strcmp(direction,'to_original'),prefix=[side 'inv'];end
M=nan(3);
for r=0:2
 for c=0:2,M(r+1,c+1)=T.(sprintf('%s%d%d',prefix,r,c))(row);end
end
assert(all(isfinite(M),'all'),'Invalid transform matrix.');
hom=[double(points)-inputBase,ones(size(points,1),1)]*M';
mappedXY=hom(:,1:2)./hom(:,3)+outputBase;
sourceFrameID0=T.source_frame0(row);
end
